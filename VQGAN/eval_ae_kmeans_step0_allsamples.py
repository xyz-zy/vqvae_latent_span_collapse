"""Step-0 evaluation using ALL training samples for k-means (not just 32 batches).

Uses all 1.28M latent vectors and 100 Lloyd iterations for better k-means fit.
"""
import json, sys, torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))
from train_small2 import ParquetImageDataset

ROOT = Path("results")
CKPT_DIR = ROOT / "ae_ckpts"
DATA_ROOT = Path(".")

ALL_AE_STEPS = [1000, 5000, 10000, 15000, 20000, 30000, 40000, 60000, 80000, 100000]
CODEBOOKS = {
    1024:  "configs/imagenet_vqgan_ref128_nodisc_cb1k_respawn.yaml",
    16384: "configs/imagenet_vqgan_ref128_nodisc_cb16k_respawn.yaml",
    65536: "configs/imagenet_vqgan_ref128_nodisc_cb65k_respawn.yaml",
}

KMEANS_ITERS = 100
CHUNK_SIZE = 4096

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--device", type=int, default=0, help="CUDA device index")
_parser.add_argument("--ae_steps", type=int, nargs="*", default=None,
                     help="subset of AE steps to evaluate (default: all)")
_cli_args = _parser.parse_args()


def to_uint8(x):
    return ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8)


def kmeans_full(z_samples, k, n_iters=100, seed=42, chunk_size=4096):
    """k-means on ALL samples with chunked distance computation."""
    N, D = z_samples.shape
    device = z_samples.device
    g = torch.Generator(device=device).manual_seed(seed)

    perm = torch.randperm(N, generator=g, device=device)[:k]
    centers = z_samples[perm].clone()
    ones_n = torch.ones(N, device=device, dtype=z_samples.dtype)

    for it in range(n_iters):
        assign = torch.empty(N, device=device, dtype=torch.long)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            d_chunk = torch.cdist(z_samples[start:end], centers)
            assign[start:end] = d_chunk.argmin(dim=1)

        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(k, device=device, dtype=z_samples.dtype)
        new_centers.index_add_(0, assign, z_samples)
        counts.index_add_(0, assign, ones_n)
        alive = counts > 0
        new_centers[alive] = new_centers[alive] / counts[alive].unsqueeze(1)
        if (~alive).any():
            n_empty = int((~alive).sum().item())
            reseed_idx = torch.randint(0, N, (n_empty,), generator=g, device=device)
            new_centers[~alive] = z_samples[reseed_idx]
        centers = new_centers

    # Final assignment for usage stats
    final_assign = torch.empty(N, device=device, dtype=torch.long)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        d_chunk = torch.cdist(z_samples[start:end], centers)
        final_assign[start:end] = d_chunk.argmin(dim=1)
    final_counts = torch.bincount(final_assign, minlength=k).float()

    return centers, final_counts


def main():
    device = torch.device(f"cuda:{_cli_args.device}")
    AE_STEPS = _cli_args.ae_steps if _cli_args.ae_steps else ALL_AE_STEPS

    split_dir = DATA_ROOT / "data" / "imagenet100_split_20k_5k"
    val_idx = np.load(split_dir / "val_idx.npy")
    train_idx = np.load(split_dir / "train_idx.npy")
    val_files = sorted((DATA_ROOT / "data" / "imagenet100_raw" / "data").glob("validation-*.parquet"))
    train_files = sorted((DATA_ROOT / "data" / "imagenet100_raw" / "data").glob("train-*.parquet"))
    val_ds = ParquetImageDataset(val_files, size=128, train=False, indices=val_idx)
    train_ds = ParquetImageDataset(train_files, size=128, train=False, indices=train_idx)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    print(f"val: {len(val_ds)}, train: {len(train_ds)}")

    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    fid.eval()
    print("precomputing real FID features...")
    with torch.no_grad():
        for batch in val_loader:
            fid.update(to_uint8(batch.to(device)), real=True)
    cached_real = {
        "sum": fid.real_features_sum.clone(),
        "cov": fid.real_features_cov_sum.clone(),
        "n": fid.real_features_num_samples.clone(),
    }
    print("done")

    results = []
    for ae_steps in AE_STEPS:
        ckpt_path = CKPT_DIR / f"step_{ae_steps:07d}_fresh.pt"
        if not ckpt_path.exists():
            print(f"skipping ae_steps={ae_steps}: not found")
            continue

        # Collect ALL encoder latents for this checkpoint
        print(f"\n--- Loading AE checkpoint step {ae_steps} ---", flush=True)
        cfg = OmegaConf.load("configs/imagenet_ae_ref128_nodisc.yaml")
        from taming.models.ae import AEModel
        mp = cfg.model.params
        ae_model = AEModel(
            ddconfig=OmegaConf.to_container(mp.ddconfig),
            lossconfig=OmegaConf.to_container(mp.lossconfig),
            embed_dim=mp.embed_dim,
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        ae_model.load_state_dict(state, strict=False)
        ae_model.eval()

        print("collecting ALL latents...", flush=True)
        all_z = []
        with torch.no_grad():
            for batch in train_loader:
                x = batch.to(device)
                h = ae_model.quant_conv(ae_model.encoder(x))
                h = h.permute(0, 2, 3, 1).reshape(-1, h.shape[1]).contiguous()
                all_z.append(h)
        z_all = torch.cat(all_z, dim=0)  # (1.28M, 256) on GPU
        print(f"  collected {z_all.shape[0]} latents", flush=True)
        del ae_model, all_z
        torch.cuda.empty_cache()

        for n_embed, cfg_path in CODEBOOKS.items():
            print(f"\n=== AE {ae_steps}, cb={n_embed}, {KMEANS_ITERS} iters, {z_all.shape[0]} samples ===", flush=True)

            # k-means on ALL samples
            centers, km_counts = kmeans_full(z_all, n_embed, n_iters=KMEANS_ITERS, chunk_size=CHUNK_SIZE)

            # Load VQModel and set codebook
            cfg = OmegaConf.load(cfg_path)
            from taming.models.vqgan import VQModel
            mp2 = cfg.model.params
            model = VQModel(
                ddconfig=OmegaConf.to_container(mp2.ddconfig),
                lossconfig=OmegaConf.to_container(mp2.lossconfig),
                n_embed=n_embed, embed_dim=mp2.embed_dim,
                vq_kwargs=OmegaConf.to_container(mp2.get("vq_kwargs", {})) or {},
            ).to(device)
            # Load AE weights
            ckpt2 = torch.load(ckpt_path, map_location=device, weights_only=False)
            state2 = ckpt2["model"] if "model" in ckpt2 else ckpt2
            model.load_state_dict(state2, strict=False)
            # Set codebook from k-means
            model.quantize.embedding.weight.data.copy_(centers.to(model.quantize.embedding.weight.dtype))
            model.eval()

            # Evaluate
            last_layer = model.decoder.conv_out.weight
            usage = torch.zeros(n_embed, dtype=torch.long, device=device)
            rec_sum = lpips_sum = mse_sum = 0.0
            n_batch = n_img = 0
            fid.fake_features_sum.zero_()
            fid.fake_features_cov_sum.zero_()
            fid.fake_features_num_samples.zero_()
            fid.real_features_sum.copy_(cached_real["sum"])
            fid.real_features_cov_sum.copy_(cached_real["cov"])
            fid.real_features_num_samples.copy_(cached_real["n"])

            with torch.no_grad():
                for batch in val_loader:
                    x = batch.to(device)
                    xrec, qloss = model(x)
                    _, log_ae = model.loss(qloss, x, xrec, 0, 0, last_layer=last_layer, split="val")
                    rec_sum += float(log_ae.get("val/rec_loss", 0))
                    lpips_sum += float(log_ae.get("val/p_loss", 0))
                    mse_sum += float(((x - xrec).float() ** 2).mean(dim=(1, 2, 3)).sum().item())
                    n_img += x.shape[0]; n_batch += 1
                    _, _, info = model.encode(x)
                    _, _, idx = info
                    usage.scatter_add_(0, idx.view(-1).long(),
                                       torch.ones(idx.view(-1).shape[0], dtype=torch.long, device=device))
                    fid.update(to_uint8(xrec), real=False)

            rec_loss = rec_sum / n_batch
            lpips = lpips_sum / n_batch
            num_used = int((usage > 0).sum().item())
            p = usage.float() / max(float(usage.sum().item()), 1.0)
            nz = p[p > 0]
            H_bits = float(-(nz * torch.log2(nz)).sum().item()) if nz.numel() else 0.0
            w = model.quantize.embedding.weight.data.float()
            w_c = w - w.mean(dim=0, keepdim=True)
            _, S, _ = torch.linalg.svd(w_c, full_matrices=False)
            var = S ** 2; cumvar = var.cumsum(0) / var.sum()
            eff_dim = int((cumvar < 0.99).sum().item()) + 1
            rfid = float(fid.compute().item())

            row = {
                "ae_steps": ae_steps, "n_embed": n_embed,
                "rec_loss": round(rec_loss, 4), "l1": round(rec_loss - lpips, 4),
                "lpips": round(lpips, 4), "mse": round(mse_sum / n_img, 5),
                "codebook_used": num_used, "util_pct": round(num_used / n_embed * 100, 1),
                "entropy_bits": round(H_bits, 2), "eff_dim_99": eff_dim,
                "rfid": round(rfid, 2),
                "kmeans_iters": KMEANS_ITERS, "kmeans_samples": int(z_all.shape[0]),
            }
            results.append(row)
            print(f"  rec={row['rec_loss']:.4f} l1={row['l1']:.4f} lpips={row['lpips']:.4f} "
                  f"used={num_used}/{n_embed} ({row['util_pct']}%) dim={eff_dim} rFID={rfid:.2f}",
                  flush=True)

            del model
            torch.cuda.empty_cache()

        del z_all
        torch.cuda.empty_cache()

    out = ROOT / f"ae_kmeans_step0_eval_allsamples_gpu{_cli_args.device}.jsonl"
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
