"""Small VQGAN trainer: modern-PyTorch port of taming-transformers' main.py.

Single-process or torchrun-based DDP. Drops all pytorch-lightning 1.0 dependencies.
Reuses the repo's VQModel (encoder/decoder/quantizer) and VQLPIPSWithDiscriminator loss.

Usage:
  Single GPU:
    python train_small.py --config configs/imagenet_vqgan_small128.yaml
  4-GPU DDP:
    torchrun --standalone --nproc_per_node=4 train_small.py --config configs/imagenet_vqgan_small128.yaml
"""
import argparse
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import albumentations as A
import pyarrow.parquet as pq
from torchvision.utils import make_grid, save_image

# Must match taming's import path; main.py provides instantiate_from_config.
from main import instantiate_from_config


def to_uint8_minus1_to_1(x):
    """Convert a float tensor in [-1, 1] to uint8 in [0, 255] for FID."""
    return ((x + 1.0) * 127.5).clamp(0.0, 255.0).to(torch.uint8)


# ---------------------------- DDP helpers ----------------------------

def ddp_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        return rank, world, local_rank, True
    torch.cuda.set_device(0)
    return 0, 1, 0, False


def is_main(rank):
    return rank == 0


# ---------------------------- Dataset ----------------------------

class ParquetImageDataset(Dataset):
    """Reads HF-style parquet shards (column `image` = struct<bytes, path>)
    and yields CHW float32 tensors normalized to [-1, 1], shape (3, size, size).

    Image bytes stay in the Arrow table (memory-mapped); only the bytes for
    __getitem__'s row are decoded.
    """
    def __init__(self, parquet_files, size=128, train=True, indices=None):
        import pyarrow as pa
        self.size = size
        tables = [pq.read_table(str(f), columns=["image"]) for f in parquet_files]
        self.table = pa.concat_tables(tables)
        self._image_col = self.table.column("image")
        if indices is not None:
            self._map = np.asarray(indices, dtype=np.int64)
            self._len = int(self._map.shape[0])
        else:
            self._map = None
            self._len = int(self.table.num_rows)

        augs = [A.SmallestMaxSize(max_size=size)]
        if train:
            augs += [A.RandomCrop(height=size, width=size), A.HorizontalFlip(p=0.5)]
        else:
            augs += [A.CenterCrop(height=size, width=size)]
        self.aug = A.Compose(augs)

    def __len__(self):
        return self._len

    def __getitem__(self, i):
        row = int(self._map[i]) if self._map is not None else i
        rec = self._image_col[row].as_py()
        img_bytes = rec["bytes"] if isinstance(rec, dict) else rec
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        arr = self.aug(image=arr)["image"]
        arr = (arr / 127.5 - 1.0).astype(np.float32)                # HWC [-1,1]
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()     # CHW
        return t


class _OffsetSamplerWrap:
    """Wraps a DistributedSampler / Sampler; on the first __iter__ call, skips the
    first `offset` indices the wrapped sampler would have produced. Subsequent epochs
    are normal. Used for strict-resume to align the DataLoader to the saved
    batch_in_epoch without re-iterating items (which would consume augmentation RNG
    via __getitem__ calls and break bit-exact resume)."""
    def __init__(self, base_sampler, offset=0):
        self.base = base_sampler
        self._pending_offset = int(offset)
        # batch_size is needed because we count BATCHES, not individual indices.
        # Set by the caller right after construction.
        self.items_per_batch = 1

    def set_epoch(self, e):
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(e)

    def __iter__(self):
        all_idx = list(iter(self.base))
        if self._pending_offset:
            skip_items = self._pending_offset * self.items_per_batch
            all_idx = all_idx[skip_items:]
            self._pending_offset = 0
        return iter(all_idx)

    def __len__(self):
        return max(0, len(self.base) - self._pending_offset * self.items_per_batch)


def build_loaders(data_root, size, batch_size, num_workers, world, rank, train=True,
                  split_dir=None):
    data_root = Path(data_root)
    train_files = sorted(data_root.glob("data/train-*.parquet"))
    val_files = sorted(data_root.glob("data/validation-*.parquet"))
    assert train_files and val_files, f"no parquet shards under {data_root}/data/"

    train_idx = val_idx = None
    if split_dir is not None:
        split_dir = Path(split_dir)
        train_idx = np.load(split_dir / "train_idx.npy")
        val_idx = np.load(split_dir / "val_idx.npy")

    train_ds = ParquetImageDataset(train_files, size=size, train=True, indices=train_idx)
    val_ds = ParquetImageDataset(val_files, size=size, train=False, indices=val_idx)

    if world > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank,
                                           shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world, rank=rank,
                                         shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=num_workers,
        pin_memory=True, drop_last=True, persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, sampler=val_sampler,
        shuffle=False, num_workers=max(1, num_workers // 2),
        pin_memory=True, drop_last=False, persistent_workers=(num_workers > 0),
    )
    return train_ds, val_ds, train_loader, val_loader, train_sampler


# ---------------------------- Effective-dim probe ----------------------------

@torch.no_grad()
def effective_dim(z, var_thresh=0.99):
    """Smallest k such that the top-k PCA components cover `var_thresh` of variance.

    `z` is the encoder latent. Shape [B, C, H, W] is flattened to [B*H*W, C] so we're
    measuring the intrinsic dim of the per-location C-dim embedding distribution, which
    is what the decoder sees at each spatial slot. Returns an int in [1, C].
    """
    if z.dim() == 4:
        B, C, H, W = z.shape
        z = z.permute(0, 2, 3, 1).reshape(B * H * W, C)
    elif z.dim() != 2:
        z = z.reshape(z.shape[0], -1)
    z = z.float()
    z = z - z.mean(dim=0, keepdim=True)
    N = z.shape[0]
    if N < 2:
        return 0
    s = torch.linalg.svdvals(z)
    variances = (s ** 2) / max(N - 1, 1)
    total = variances.sum()
    if total <= 0:
        return 0
    cum = torch.cumsum(variances / total, dim=0)
    k = int((cum < var_thresh).sum().item()) + 1
    return k


# ---------------------------- Image logging ----------------------------

def save_recon_grid(xrec, x, path, max_n=8):
    n = min(x.size(0), max_n)
    x = x[:n].clamp(-1, 1)
    xrec = xrec[:n].clamp(-1, 1)
    grid = torch.cat([x, xrec], dim=0)            # top row real, bottom recon
    grid = (grid + 1.0) / 2.0
    save_image(make_grid(grid, nrow=n), path)


# ---------------------------- Main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--data_root", type=str,
                    default="data/imagenet100_raw")
    ap.add_argument("--log_root", type=str,
                    default="logs")
    ap.add_argument("--run_name", type=str, default="small128")
    ap.add_argument("--max_steps", type=int, default=60000)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--image_every", type=int, default=500)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--ckpt_steps", type=int, nargs="*", default=[],
                    help="additional specific steps at which to save checkpoints")
    ap.add_argument("--eff_dim_every", type=int, default=100)
    ap.add_argument("--beta", type=float, default=None,
                    help="override VQ commitment loss beta (default: use model's 0.25)")
    ap.add_argument("--split_dir", type=str, default=None,
                    help="directory with train_idx.npy and val_idx.npy to restrict rows")
    ap.add_argument("--fid_every_val", type=int, default=1,
                    help="compute rFID every Nth val call (1 = every val; higher to save cost)")
    ap.add_argument("--kmeans_init_batches", type=int, default=0,
                    help="if >0, run k-means codebook init on this many batches of encoder outputs before training")
    ap.add_argument("--kmeans_iters", type=int, default=20,
                    help="number of Lloyd's iterations for k-means init")
    ap.add_argument("--resume_from", type=str, default=None,
                    help="path to checkpoint .pt file; loads model weights (NOT optimizer) via strict=False. "
                         "Combine with --strict_resume for full-state resume.")
    ap.add_argument("--strict_resume", action="store_true",
                    help="when set with --resume_from, also restore optimizer state + step/epoch + EMA "
                         "(requires identical architecture); skips k-means init and the step-0 probe")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--lr_warmup_steps", type=int, default=0,
                    help="linear warmup from 0 to peak over this many steps; 0 disables warmup")
    ap.add_argument("--lr_decay_total_steps", type=int, default=0,
                    help="total steps over which cosine decay runs (from step 0 to this step); "
                         "0 disables decay (constant peak LR)")
    ap.add_argument("--lr_min_ratio", type=float, default=0.1,
                    help="decay floor as a fraction of peak LR (e.g. 0.1 = cosine from 1.0 to 0.1)")
    args = ap.parse_args()
    ckpt_steps_set = set(args.ckpt_steps)

    rank, world, local_rank, ddp = ddp_setup()
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    print(f"[rank{rank}] ddp_setup done, device={device} world={world}", flush=True)

    cfg = OmegaConf.load(args.config)
    size = cfg.model.params.ddconfig.resolution
    per_gpu_bs = cfg.data.params.batch_size
    base_lr = float(cfg.model.base_learning_rate)

    # taming's main.py scales lr = accumulate * ngpu * bs * base_lr
    lr = base_lr * world * per_gpu_bs
    if is_main(rank):
        print(f"[cfg] world={world} per_gpu_bs={per_gpu_bs} size={size} base_lr={base_lr} lr={lr:.2e}")

    # ---- data ----
    train_ds, val_ds, train_loader, val_loader, train_sampler = build_loaders(
        args.data_root, size=size, batch_size=per_gpu_bs,
        num_workers=args.num_workers, world=world, rank=rank,
        split_dir=args.split_dir,
    )
    if is_main(rank):
        print(f"[data] train={len(train_ds)} val={len(val_ds)} "
              f"steps/epoch={len(train_loader)} "
              f"(split_dir={args.split_dir})")

    # ---- model ----
    print(f"[rank{rank}] instantiating model", flush=True)
    model = instantiate_from_config(cfg.model).to(device)
    last_layer = model.decoder.conv_out.weight
    print(f"[rank{rank}] model on {device}", flush=True)

    ae_params = (list(model.encoder.parameters())
                 + list(model.decoder.parameters())
                 + list(model.quantize.parameters())
                 + list(model.quant_conv.parameters())
                 + list(model.post_quant_conv.parameters()))
    disc_params = list(model.loss.discriminator.parameters())
    opt_ae = torch.optim.Adam(ae_params, lr=lr, betas=(0.5, 0.9))
    opt_disc = torch.optim.Adam(disc_params, lr=lr, betas=(0.5, 0.9))

    # Shared LR schedule: linear warmup from 0 to peak, then cosine decay from
    # peak to `lr_min_ratio * peak` at step=lr_decay_total_steps. Defaults
    # (warmup=0, decay_total=0) make _lr_lambda return 1.0 for all steps, which
    # keeps behavior identical to pre-scheduler runs.
    def _lr_lambda(step: int) -> float:
        if args.lr_warmup_steps > 0 and step < args.lr_warmup_steps:
            return (step + 1) / max(1, args.lr_warmup_steps)
        if args.lr_decay_total_steps > 0:
            post_warmup = step - args.lr_warmup_steps
            total = max(1, args.lr_decay_total_steps - args.lr_warmup_steps)
            progress = min(1.0, max(0.0, post_warmup / total))
            return (args.lr_min_ratio
                    + (1.0 - args.lr_min_ratio) * 0.5
                      * (1.0 + math.cos(math.pi * progress)))
        return 1.0
    sched_ae   = LambdaLR(opt_ae,   lr_lambda=_lr_lambda)
    sched_disc = LambdaLR(opt_disc, lr_lambda=_lr_lambda)

    if ddp:
        # The AE-phase backward computes grads on disc params (via g_loss) and the
        # disc-phase backward then overwrites those grads -> default DDP rejects this
        # as "variable marked ready twice". static_graph=True tells DDP to compute
        # the set of used params once from the first iteration and reuse the plan.
        print(f"[rank{rank}] wrapping DDP", flush=True)
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False,
                    find_unused_parameters=False, static_graph=True)
        print(f"[rank{rank}] DDP wrapped", flush=True)
        get_module = lambda: model.module
    else:
        get_module = lambda: model

    n_params = sum(p.numel() for p in get_module().parameters())
    if args.beta is not None:
        quantize = getattr(get_module(), "quantize", None)
        if quantize is not None and hasattr(quantize, "beta"):
            quantize.beta = args.beta
            if is_main(rank):
                print(f"[beta] overridden to {args.beta}")
    if is_main(rank):
        print(f"[model] total params = {n_params/1e6:.2f}M")

    # ---- log dir ----
    log_dir = Path(args.log_root) / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.run_name}"
    if is_main(rank):
        (log_dir / "images").mkdir(parents=True, exist_ok=True)
        (log_dir / "ckpts").mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, log_dir / "config.yaml")
        print(f"[logdir] {log_dir}")
    if ddp:
        dist.barrier()

    # Header note: train_small2.py is the bit-faithful-strict-resume version.
    # Use --strict_resume + --resume_from PATH to continue a run as if uninterrupted.
    # For weights-only warmstart (e.g., AE -> VQ where the param set differs),
    # use train_small.py (the simpler non-resumable variant).
    metrics_fp = None
    if is_main(rank):
        metrics_fp = open(log_dir / "metrics.jsonl", "a")

    latest_val_rec = None  # resume path may overwrite; training loop reads it
    best_val_rec_loss = float("inf")  # tracker for "best" ckpt; restored on strict-resume

    # ---- Resume from checkpoint (optional; runs before kmeans init) ----
    # Two modes:
    #  * default (--resume_from PATH): load model weights with strict=False.
    #    Good for architecture-changing warmstarts (e.g. AE -> VQ: VQ has extra
    #    quantize.embedding + my respawn buffers; AE has neither). Fresh optimizer.
    #  * strict (--resume_from PATH --strict_resume): also restore both optimizer
    #    state dicts + the saved step + epoch. Requires identical architecture.
    #    k-means init and the step-0 probe are skipped so we continue exactly
    #    from the saved trajectory.
    resume_step = 0
    resume_epoch = 0
    if args.resume_from:
        if is_main(rank):
            mode = "strict (full state)" if args.strict_resume else "weights-only"
            print(f"[resume] loading {args.resume_from} ({mode})", flush=True)
        ckpt_raw = torch.load(args.resume_from, map_location=device, weights_only=False)
        state = ckpt_raw.get("model", ckpt_raw) if isinstance(ckpt_raw, dict) else ckpt_raw
        missing, unexpected = get_module().load_state_dict(
            state, strict=args.strict_resume)
        if is_main(rank):
            print(f"[resume] model loaded (strict={args.strict_resume}); "
                  f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
            if missing and not args.strict_resume:
                print(f"[resume]   missing sample: {missing[:4]}", flush=True)
            if unexpected:
                print(f"[resume]   unexpected sample: {unexpected[:4]}", flush=True)

        if args.strict_resume:
            if not isinstance(ckpt_raw, dict):
                raise RuntimeError("--strict_resume requires a dict-format checkpoint "
                                   "(with 'opt_ae', 'opt_disc', 'step' fields); "
                                   "this file looks like a bare state_dict.")
            if "opt_ae" not in ckpt_raw or "opt_disc" not in ckpt_raw:
                raise RuntimeError("--strict_resume: checkpoint is missing 'opt_ae' "
                                   "or 'opt_disc'; cannot restore optimizer state.")
            opt_ae.load_state_dict(ckpt_raw["opt_ae"])
            opt_disc.load_state_dict(ckpt_raw["opt_disc"])
            # Restore LR scheduler state if present; fall back to fast-forwarding
            # the scheduler for ckpts saved before the scheduler feature landed.
            if "sched_ae" in ckpt_raw:
                sched_ae.load_state_dict(ckpt_raw["sched_ae"])
            if "sched_disc" in ckpt_raw:
                sched_disc.load_state_dict(ckpt_raw["sched_disc"])
            if "sched_ae" not in ckpt_raw and int(ckpt_raw.get("step", 0)) > 0:
                n = int(ckpt_raw["step"])
                for _ in range(n):
                    sched_ae.step(); sched_disc.step()
                if is_main(rank):
                    print(f"[strict-resume] scheduler state not in ckpt; "
                          f"fast-forwarded by {n} steps to align LR", flush=True)
            resume_step = int(ckpt_raw.get("step", 0))
            resume_epoch = int(ckpt_raw.get("epoch", 0))
            resume_batch_in_epoch = int(ckpt_raw.get("batch_in_epoch", 0))
            latest_val_rec = ckpt_raw.get("val_rec_loss", None)
            if latest_val_rec is not None:
                try:
                    latest_val_rec = float(latest_val_rec)
                except (TypeError, ValueError):
                    latest_val_rec = None
            saved_best = ckpt_raw.get("best_val_rec_loss", None)
            if isinstance(saved_best, (int, float)):
                best_val_rec_loss = float(saved_best)
            # Restore log-EMA smoothing dict (cosmetic — keeps the train log line continuous)
            saved_running = ckpt_raw.get("running")
            if isinstance(saved_running, dict):
                running_resume = {k: float(v) for k, v in saved_running.items()
                                  if isinstance(v, (int, float))}
            else:
                running_resume = None
            # Restore RNG states. Each rank loads its own CUDA RNG entry from the
            # per-rank list saved at ckpt time. Older ckpts without 'rng' get fresh RNGs.
            rng = ckpt_raw.get("rng")
            if isinstance(rng, dict):
                if "python" in rng:
                    random.setstate(rng["python"])
                if "numpy" in rng:
                    np.random.set_state(rng["numpy"])
                if "torch_cpu" in rng:
                    # set_rng_state requires a CPU ByteTensor; map_location may have
                    # moved it to GPU during torch.load.
                    cpu_state = rng["torch_cpu"].cpu().to(torch.uint8)
                    torch.set_rng_state(cpu_state)
                cuda_list = rng.get("torch_cuda_per_rank")
                if isinstance(cuda_list, (list, tuple)) and rank < len(cuda_list):
                    cuda_state = cuda_list[rank].cpu().to(torch.uint8)
                    torch.cuda.set_rng_state(cuda_state, device=local_rank)
                if is_main(rank):
                    print(f"[strict-resume] restored RNG states "
                          f"(python/numpy/torch_cpu/torch_cuda × {len(cuda_list) if cuda_list else 0} ranks)",
                          flush=True)
            else:
                running_resume = running_resume  # noop; just for symmetry
            if is_main(rank):
                print(f"[strict-resume] restored optimizers + step={resume_step} "
                      f"epoch={resume_epoch} batch_in_epoch={resume_batch_in_epoch} "
                      f"latest_val_rec={latest_val_rec}", flush=True)
            if args.kmeans_init_batches > 0 and is_main(rank):
                print("[strict-resume] ignoring --kmeans_init_batches "
                      "(codebook already trained)", flush=True)
            args.kmeans_init_batches = 0
        else:
            resume_batch_in_epoch = 0
            running_resume = None
        del ckpt_raw, state
        torch.cuda.empty_cache()
    else:
        resume_batch_in_epoch = 0
        running_resume = None

    # ---- Bit-exact resume: wrap sampler so the first post-resume epoch starts at
    # the saved batch offset without iterating skipped items (which would consume
    # augmentation RNG and break bit-exactness). Rebuilds train_loader with the
    # wrapped sampler.
    if args.strict_resume and resume_batch_in_epoch > 0:
        if train_sampler is None:
            # Single-GPU path: construct an explicit RandomSampler to wrap. Note that
            # RandomSampler uses the *global* torch RNG for shuffling, so the resumed
            # shuffle order will diverge from the uninterrupted run's (which used the
            # original torch RNG). Bit-exactness in this branch is approximate.
            train_sampler = torch.utils.data.RandomSampler(train_ds)
        wrapped = _OffsetSamplerWrap(train_sampler, offset=resume_batch_in_epoch)
        wrapped.items_per_batch = per_gpu_bs
        train_sampler = wrapped
        # Rebuild train_loader so it picks up the new (wrapped) sampler.
        nw = args.num_workers
        train_loader = DataLoader(
            train_ds, batch_size=per_gpu_bs, sampler=train_sampler,
            shuffle=False, num_workers=nw, pin_memory=True, drop_last=True,
            persistent_workers=(nw > 0),
        )
        if is_main(rank):
            print(f"[strict-resume] rebuilt train_loader with offset-wrapped sampler "
                  f"(skip {resume_batch_in_epoch} batches on epoch {resume_epoch})",
                  flush=True)

    # ---- K-means codebook init (optional; runs before training) ----
    quantize_mod = getattr(get_module(), "quantize", None)
    if (args.kmeans_init_batches > 0
            and quantize_mod is not None
            and hasattr(quantize_mod, "kmeans_init_codebook")):
        if is_main(rank):
            print(f"[kmeans-init] collecting {args.kmeans_init_batches} batches of "
                  f"encoder outputs...", flush=True)
        get_module().eval()
        samples = []
        loader_iter = iter(train_loader)
        for _ in range(args.kmeans_init_batches):
            batch = next(loader_iter).to(device, non_blocking=True)
            with torch.no_grad():
                h = get_module().quant_conv(get_module().encoder(batch))
                # h: (B, e_dim, H, W) -> (B*H*W, e_dim)
                h = h.permute(0, 2, 3, 1).reshape(-1, h.shape[1]).contiguous()
            samples.append(h)
        local_samples = torch.cat(samples, dim=0)
        if ddp and world > 1:
            gather_list = [torch.empty_like(local_samples) for _ in range(world)]
            dist.all_gather(gather_list, local_samples)
            z_samples = torch.cat(gather_list, dim=0)
        else:
            z_samples = local_samples
        if is_main(rank):
            print(f"[kmeans-init] running k-means on {z_samples.shape[0]} samples, "
                  f"k={quantize_mod.n_e}, iters={args.kmeans_iters}...", flush=True)
        # Identical seed across ranks -> bit-identical centers on every rank.
        quantize_mod.kmeans_init_codebook(
            z_samples, n_iters=args.kmeans_iters, seed=args.seed,
            n_batches=args.kmeans_init_batches)
        if is_main(rank):
            w = quantize_mod.embedding.weight.data
            print(f"[kmeans-init] done. codebook mean={w.mean().item():.4f} "
                  f"std={w.std().item():.4f} abs-mean={w.abs().mean().item():.4f}",
                  flush=True)
        get_module().train()
        del samples, local_samples, z_samples
        torch.cuda.empty_cache()

    # ---- FID setup: precompute real Inception features once ----
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    if is_main(rank):
        print("[fid] precomputing real inception features on val set...", flush=True)
    fid.eval()
    with torch.no_grad():
        for batch in val_loader:
            x = batch.to(device, non_blocking=True)
            fid.update(to_uint8_minus1_to_1(x), real=True)
    cached_real = {
        "sum": fid.real_features_sum.detach().clone(),
        "cov": fid.real_features_cov_sum.detach().clone(),
        "n":   fid.real_features_num_samples.detach().clone(),
    }
    if is_main(rank):
        print(f"[fid] cached real features (this rank's n={int(cached_real['n'].item())})", flush=True)

    # ---- init (step 0) effective-dim baseline ----
    # Probe the randomly initialized encoder + codebook once, so later plots have
    # a pre-training reference point. Uses one real training batch.
    # NOTE: only rank 0 runs this. The quantizer has DDP collectives inside its
    # training-mode forward (hits all_reduce), so we put the quantizer in eval
    # mode for this rank-0-only call to avoid deadlocking the other ranks.
    # Skipped when strict-resuming (we're picking up where a run left off).
    if is_main(rank) and not args.strict_resume:
        init_batch = next(iter(train_loader)).to(device, non_blocking=True)
        quantize_mod_ref = getattr(get_module(), "quantize", None)
        was_training = (quantize_mod_ref is not None and quantize_mod_ref.training)
        if was_training:
            quantize_mod_ref.eval()
        with torch.no_grad():
            z_init = get_module().encode(init_batch)
            if isinstance(z_init, tuple):
                z_init = z_init[0]
        if was_training:
            quantize_mod_ref.train()
        init_eff = effective_dim(z_init, var_thresh=0.99)
        msg = f"[init] latent eff_dim_99 = {init_eff}"
        record = {"step": 0, "train/eff_dim_99": float(init_eff)}
        quantize = getattr(get_module(), "quantize", None)
        if quantize is not None and hasattr(quantize, "embedding"):
            init_cb = effective_dim(quantize.embedding.weight.detach(), var_thresh=0.99)
            msg += f"  codebook eff_dim_99 = {init_cb}"
            record["train/codebook_eff_dim_99"] = float(init_cb)
        print(msg, flush=True)
        metrics_fp.write(json.dumps(record) + "\n")
        metrics_fp.flush()
        del init_batch, z_init

    # ---- train loop ----
    step = resume_step
    epoch = resume_epoch
    t_last = time.time()
    t_epoch_start = time.time()
    running = dict(running_resume) if running_resume is not None else {}
    # latest_val_rec carried over from resume if --strict_resume, else stays None
    done = False
    # best-checkpoint tracker is initialized above the resume block; restored from
    # checkpoint in --strict_resume mode if the field is present.
    if args.strict_resume and is_main(rank):
        print(f"[strict-resume] continuing training from step={step} epoch={epoch} "
              f"batch_in_epoch={resume_batch_in_epoch}", flush=True)
    while not done:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if is_main(rank):
            print(f"[epoch {epoch}] start at step {step}", flush=True)
        # batch position counter, snapshotted into ckpt; reset at each epoch boundary.
        # On the first post-resume epoch, _OffsetSamplerWrap (if present) advances the
        # sampler past the first N batches without iterating items — so __getitem__
        # never fires for skipped indices and augmentation RNG isn't consumed. The
        # wrapper resets itself after the first iter; we still bump batch_in_epoch
        # to reflect the saved offset.
        batch_in_epoch = resume_batch_in_epoch
        resume_batch_in_epoch = 0
        for batch in train_loader:
            x = batch.to(device, non_blocking=True)

            # ---- AE phase ----
            xrec, qloss = model(x)
            aeloss, log_ae = get_module().loss(
                qloss, x, xrec, 0, step,
                last_layer=last_layer, split="train",
            )
            opt_ae.zero_grad(set_to_none=True)
            opt_disc.zero_grad(set_to_none=True)
            aeloss.backward()
            opt_ae.step()
            sched_ae.step()

            # ---- Disc phase ---- (disc step detaches internally)
            discloss, log_disc = get_module().loss(
                qloss.detach(), x, xrec.detach(), 1, step,
                last_layer=last_layer, split="train",
            )
            opt_ae.zero_grad(set_to_none=True)
            opt_disc.zero_grad(set_to_none=True)
            discloss.backward()
            opt_disc.step()
            sched_disc.step()

            # ---- logging ----
            for k, v in {**log_ae, **log_disc}.items():
                vv = float(v.detach().item()) if torch.is_tensor(v) else float(v)
                running[k] = running.get(k, 0.0) * 0.98 + vv * 0.02

            step += 1
            batch_in_epoch += 1

            # ---- effective-dim probes (latent and codebook) ----
            # Only rank 0 runs this; we put the quantizer in eval mode for the
            # duration of the encode() call so its training-mode DDP collectives
            # (hits all_reduce / optional broadcast) don't fire and deadlock the
            # other ranks.
            if is_main(rank) and args.eff_dim_every > 0 and step % args.eff_dim_every == 0:
                _q = getattr(get_module(), "quantize", None)
                _was_tr = (_q is not None and _q.training)
                if _was_tr:
                    _q.eval()
                with torch.no_grad():
                    z_probe = get_module().encode(x)
                    if isinstance(z_probe, tuple):   # VQModel.encode returns (quant, emb_loss, info)
                        z_probe = z_probe[0]
                if _was_tr:
                    _q.train()
                eff99 = effective_dim(z_probe, var_thresh=0.99)
                running["train/eff_dim_99"] = float(eff99)
                # Mean L2 norm of encoder latents, per token. Reshape
                # (B,C,H,W) -> (B*H*W, C) and take row norms.
                if z_probe.dim() == 4:
                    _B, _C, _H, _W = z_probe.shape
                    _flat = z_probe.permute(0, 2, 3, 1).reshape(-1, _C)
                else:
                    _flat = z_probe.reshape(z_probe.shape[0], -1)
                running["train/latent_l2_mean"] = float(
                    _flat.float().norm(dim=1).mean().item())
                quantize = getattr(get_module(), "quantize", None)
                if quantize is not None and hasattr(quantize, "embedding"):
                    cb = quantize.embedding.weight
                    cb_eff = effective_dim(cb.detach(), var_thresh=0.99)
                    running["train/codebook_eff_dim_99"] = float(cb_eff)
                    # Mean L2 norm per codebook entry.
                    running["train/codebook_l2_mean"] = float(
                        cb.detach().float().norm(dim=1).mean().item())
                    respawned = getattr(quantize, "last_n_respawned", None)
                    if respawned is not None:
                        running["train/codebook_respawned"] = float(respawned.item())

            if is_main(rank) and step % args.log_every == 0:
                dt = time.time() - t_last
                ips = args.log_every * per_gpu_bs * world / max(dt, 1e-6)
                running["train/lr"] = float(sched_ae.get_last_lr()[0])
                msg = (f"step {step:>7d} | {ips:6.1f} img/s | "
                       f"rec {running.get('train/rec_loss', 0):.4f} | "
                       f"nll {running.get('train/nll_loss', 0):.4f} | "
                       f"p {running.get('train/p_loss', 0):.4f} | "
                       f"q {running.get('train/quant_loss', 0):.4f} | "
                       f"g {running.get('train/g_loss', 0):.4f} | "
                       f"d {running.get('train/disc_loss', 0):.4f} | "
                       f"dw {running.get('train/d_weight', 0):.3f} | "
                       f"effdim99 {running.get('train/eff_dim_99', 0):.0f} | "
                       f"cbeff99 {running.get('train/codebook_eff_dim_99', 0):.0f}")
                print(msg, flush=True)
                if metrics_fp is not None:
                    metrics_fp.write(json.dumps({"step": step, "ips": ips, **running}) + "\n")
                    metrics_fp.flush()
                t_last = time.time()

            if is_main(rank) and step % args.image_every == 0:
                with torch.no_grad():
                    save_recon_grid(xrec, x, log_dir / "images" / f"train_{step:07d}.png")

            if step % args.val_every == 0:
                val_call_idx = step // args.val_every
                do_fid = (args.fid_every_val > 0 and
                          val_call_idx % args.fid_every_val == 0)
                val_agg = validate(get_module(), val_loader, device, last_layer, log_dir,
                                   step, rank, world, is_main(rank), metrics_fp,
                                   fid=fid, cached_real=cached_real, do_fid=do_fid)
                if val_agg is not None and "val/rec_loss" in val_agg:
                    latest_val_rec = float(val_agg["val/rec_loss"])
                    # Best-ckpt save — every rank computed the same val_rec_loss
                    # (validate() all_reduces sums), so the "is this a new best"
                    # decision is consistent across ranks. The CUDA-RNG all_gather
                    # is a collective and must run on all ranks when invoked.
                    if latest_val_rec < best_val_rec_loss:
                        best_val_rec_loss = latest_val_rec
                        if ddp:
                            local_cuda_rng = torch.cuda.get_rng_state(local_rank).to(device)
                            gathered = [torch.empty_like(local_cuda_rng) for _ in range(world)]
                            dist.all_gather(gathered, local_cuda_rng)
                            cuda_rng_per_rank = [g.cpu() for g in gathered]
                        else:
                            cuda_rng_per_rank = [torch.cuda.get_rng_state(0)]
                        if is_main(rank):
                            best_ckpt = {
                                "step": step,
                                "epoch": epoch,
                                "batch_in_epoch": batch_in_epoch,
                                "val_rec_loss": latest_val_rec,
                                "best_val_rec_loss": best_val_rec_loss,
                                "model": get_module().state_dict(),
                                "opt_ae": opt_ae.state_dict(),
                                "opt_disc": opt_disc.state_dict(),
                                "sched_ae": sched_ae.state_dict(),
                                "sched_disc": sched_disc.state_dict(),
                                "running": {k: float(v) for k, v in running.items()
                                            if isinstance(v, (int, float))},
                                "rng": {
                                    "python": random.getstate(),
                                    "numpy": np.random.get_state(),
                                    "torch_cpu": torch.get_rng_state(),
                                    "torch_cuda_per_rank": cuda_rng_per_rank,
                                },
                                "config": OmegaConf.to_container(cfg),
                            }
                            torch.save(best_ckpt, log_dir / "ckpts" / "best.pt")
                            print(f"[best] new best val/rec_loss={best_val_rec_loss:.4f} "
                                  f"@ step {step} -> saved best.pt", flush=True)

            if step % args.ckpt_every == 0 or step in ckpt_steps_set:
                # All ranks gather their CUDA RNG state (collective; must run on every rank).
                if ddp:
                    local_cuda_rng = torch.cuda.get_rng_state(local_rank).to(device)
                    gathered = [torch.empty_like(local_cuda_rng) for _ in range(world)]
                    dist.all_gather(gathered, local_cuda_rng)
                    cuda_rng_per_rank = [g.cpu() for g in gathered]
                else:
                    cuda_rng_per_rank = [torch.cuda.get_rng_state(0)]

                if is_main(rank):
                    ckpt = {
                        "step": step,
                        "epoch": epoch,
                        "batch_in_epoch": batch_in_epoch,
                        "val_rec_loss": latest_val_rec,
                        "best_val_rec_loss": best_val_rec_loss,
                        "model": get_module().state_dict(),
                        "opt_ae": opt_ae.state_dict(),
                        "opt_disc": opt_disc.state_dict(),
                        "sched_ae": sched_ae.state_dict(),
                        "sched_disc": sched_disc.state_dict(),
                        "running": {k: float(v) for k, v in running.items()
                                    if isinstance(v, (int, float))},
                        "rng": {
                            "python": random.getstate(),
                            "numpy": np.random.get_state(),
                            "torch_cpu": torch.get_rng_state(),
                            "torch_cuda_per_rank": cuda_rng_per_rank,
                        },
                        "config": OmegaConf.to_container(cfg),
                    }
                    valrec_str = (f"{latest_val_rec:.4f}"
                                  if latest_val_rec is not None else "na")
                    ckpt_name = f"step_{step:07d}_valrec_{valrec_str}.pt"
                    torch.save(ckpt, log_dir / "ckpts" / ckpt_name)
                    torch.save(ckpt, log_dir / "ckpts" / "last.pt")

            if step >= args.max_steps:
                done = True
                break
        if is_main(rank):
            dt_epoch = time.time() - t_epoch_start
            print(f"[epoch {epoch}] done: {dt_epoch:.1f}s "
                  f"({len(train_loader)*per_gpu_bs*world/dt_epoch:.0f} img/s)",
                  flush=True)
        t_epoch_start = time.time()
        epoch += 1

    if is_main(rank):
        if metrics_fp is not None:
            metrics_fp.close()
        print("[done]")
    if ddp:
        dist.destroy_process_group()


@torch.no_grad()
def validate(model, val_loader, device, last_layer, log_dir, step, rank, world, main,
             metrics_fp, fid=None, cached_real=None, do_fid=False):
    model.eval()
    module = model.module if hasattr(model, "module") else model
    quantize = getattr(module, "quantize", None)
    has_codebook = quantize is not None and hasattr(quantize, "embedding")
    if has_codebook:
        n_embed = int(quantize.n_e)
        usage = torch.zeros(n_embed, dtype=torch.long, device=device)

    if do_fid and fid is not None:
        # Zero fake accumulators, restore cached real accumulators.
        fid.fake_features_sum.zero_()
        fid.fake_features_cov_sum.zero_()
        fid.fake_features_num_samples.zero_()
        fid.real_features_sum.copy_(cached_real["sum"])
        fid.real_features_cov_sum.copy_(cached_real["cov"])
        fid.real_features_num_samples.copy_(cached_real["n"])

    sums = {}
    mse_sum = torch.zeros((), device=device)
    mse_n = torch.zeros((), device=device)
    n = 0
    first_x, first_xrec = None, None
    for batch in val_loader:
        x = batch.to(device, non_blocking=True)
        xrec, qloss = model(x)
        _, log_ae = model.loss(qloss, x, xrec, 0, step, last_layer=last_layer, split="val")
        _, log_disc = model.loss(qloss, x, xrec, 1, step, last_layer=last_layer, split="val")
        for k, v in {**log_ae, **log_disc}.items():
            vv = v.detach().float()
            sums[k] = sums.get(k, torch.zeros((), device=device)) + vv
        # per-pixel MSE on the [-1, 1] image space; reduce across pixels per batch,
        # track per-image sum so we can average over all val images (not per-batch-call).
        mse_per_image = ((x - xrec).float() ** 2).mean(dim=(1, 2, 3))   # [B]
        mse_sum += mse_per_image.sum()
        mse_n += float(mse_per_image.numel())
        n += 1
        if first_x is None:
            first_x = x.detach()
            first_xrec = xrec.detach()
        if has_codebook:
            # module.encode(x) returns (z_q, emb_loss, (perplexity, min_encodings, indices))
            _, _, info = module.encode(x)
            _, _, min_enc_idx = info
            idx = min_enc_idx.view(-1).to(torch.long)
            usage.scatter_add_(0, idx, torch.ones_like(idx))
        if do_fid and fid is not None:
            fid.update(to_uint8_minus1_to_1(xrec), real=False)
    # reduce
    if world > 1:
        nt = torch.tensor([n], device=device, dtype=torch.float32)
        dist.all_reduce(nt, op=dist.ReduceOp.SUM)
        n_total = float(nt.item())
        for k in list(sums.keys()):
            dist.all_reduce(sums[k], op=dist.ReduceOp.SUM)
        agg = {k: float(v.item()) / n_total for k, v in sums.items()}
        if has_codebook:
            dist.all_reduce(usage, op=dist.ReduceOp.SUM)
        dist.all_reduce(mse_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(mse_n, op=dist.ReduceOp.SUM)
    else:
        agg = {k: float(v.item()) / max(n, 1) for k, v in sums.items()}
    agg["val/mse"] = float((mse_sum / mse_n.clamp(min=1.0)).item())
    # LPIPS is already equal to val/p_loss but we also emit it under the explicit key.
    if "val/p_loss" in agg:
        agg["val/lpips"] = agg["val/p_loss"]
    # codebook usage stats
    if has_codebook:
        total = float(usage.sum().item())
        num_used = int((usage > 0).sum().item())
        p = usage.float() / max(total, 1.0)
        nz = p[p > 0]
        H_bits = float(-(nz * torch.log2(nz)).sum().item()) if nz.numel() else 0.0
        perplexity = float(2.0 ** H_bits)
        agg["val/codebook_used"] = num_used
        agg["val/codebook_entropy_bits"] = H_bits
        agg["val/codebook_perplexity"] = perplexity
    # rFID (computed only on val calls where do_fid=True)
    if do_fid and fid is not None:
        try:
            fid_value = float(fid.compute().item())
        except Exception as e:
            fid_value = float("nan")
            if main:
                print(f"[fid] compute failed: {e}", flush=True)
        agg["val/rfid"] = fid_value
    if main:
        msg = (f"[val] step {step} rec {agg.get('val/rec_loss', 0):.4f} "
               f"mse {agg.get('val/mse', 0):.5f} "
               f"lpips {agg.get('val/lpips', 0):.4f} "
               f"disc {agg.get('val/disc_loss', 0):.4f}")
        if "val/rfid" in agg:
            msg += f" rFID {agg['val/rfid']:.2f}"
        if has_codebook:
            msg += (f" | cb_used {agg['val/codebook_used']}/{n_embed} "
                    f"H {agg['val/codebook_entropy_bits']:.2f}bits "
                    f"ppl {agg['val/codebook_perplexity']:.1f}")
        print(msg, flush=True)
        if metrics_fp is not None:
            metrics_fp.write(json.dumps({"step": step, "val": agg}) + "\n")
            metrics_fp.flush()
        if first_x is not None:
            save_recon_grid(first_xrec, first_x, log_dir / "images" / f"val_{step:07d}.png")
    model.train()
    return agg


if __name__ == "__main__":
    import traceback
    rank = os.environ.get("RANK", "0")
    errfile = f"/tmp/trainsmall_err_rank{rank}.log"
    try:
        main()
    except BaseException as e:
        with open(errfile, "w") as f:
            f.write(f"rank={rank} type={type(e).__name__} msg={e}\n")
            traceback.print_exc(file=f)
        print(f"[rank={rank}] CRASH written to {errfile}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
