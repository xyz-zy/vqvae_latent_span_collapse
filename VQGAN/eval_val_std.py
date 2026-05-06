"""Compute per-sample mean and std of L1, LPIPS, rFID across the validation set.

For L1 and LPIPS: per-image values, then mean/std across images.
For rFID: bootstrap (resample val set N times, compute FID each time, take std).

Usage:
  python3 eval_val_std.py --run_dir /path/to/run --config /path/to/config.yaml
  python3 eval_val_std.py --run_dirs_file runs.txt   # one run_dir per line
  python3 eval_val_std.py --run_dir /path/to/run --config /path/to/config.yaml --skip_rfid
"""
import argparse
import json
import sys
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from train_small2 import ParquetImageDataset
from taming.modules.losses.lpips import LPIPS

DATA_ROOT = Path(".")
CONFIGS_DIR = DATA_ROOT / "configs"


def find_config_for_run(run_dir):
    cfg_in_run = run_dir / "config.yaml"
    if cfg_in_run.exists():
        return cfg_in_run
    name = run_dir.name
    if "cb65k" in name:
        if "vanilla" in name:
            return CONFIGS_DIR / "imagenet_vqgan_ref128_nodisc_cb65k_vanilla.yaml"
        return CONFIGS_DIR / "imagenet_vqgan_ref128_nodisc_cb65k_respawn.yaml"
    elif "cb16k" in name:
        if "vanilla" in name:
            return CONFIGS_DIR / "imagenet_vqgan_ref128_nodisc_cb16k_vanilla.yaml"
        return CONFIGS_DIR / "imagenet_vqgan_ref128_nodisc_cb16k_respawn.yaml"
    elif "cb1k" in name:
        if "vanilla" in name:
            return CONFIGS_DIR / "imagenet_vqgan_ref128_nodisc_cb1k_vanilla.yaml"
        return CONFIGS_DIR / "imagenet_vqgan_ref128_nodisc_cb1k_respawn.yaml"
    return None


def find_ckpt(run_dir):
    ckpts = run_dir / "ckpts"
    for name in ["best.pt", "last.pt"]:
        p = ckpts / name
        if p.exists():
            return p
    pts = sorted(ckpts.glob("step_*.pt"))
    return pts[-1] if pts else None


def load_model(config_path, ckpt_path, device):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(config_path)
    mp = cfg.model.params
    target = cfg.model.target

    if "AEModel" in target or "ae" in target.lower():
        from taming.models.ae import AEModel
        model = AEModel(
            ddconfig=OmegaConf.to_container(mp.ddconfig),
            lossconfig=OmegaConf.to_container(mp.lossconfig),
            embed_dim=mp.embed_dim,
        )
    else:
        from taming.models.vqgan import VQModel
        model = VQModel(
            ddconfig=OmegaConf.to_container(mp.ddconfig),
            lossconfig=OmegaConf.to_container(mp.lossconfig),
            n_embed=mp.n_embed, embed_dim=mp.embed_dim,
            vq_kwargs=OmegaConf.to_container(mp.get("vq_kwargs", {})) or {},
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def eval_run(run_dir, config_path, device, val_loader, lpips_fn, skip_rfid=False,
             fid_module=None, n_bootstrap=10):
    run_dir = Path(run_dir)
    ckpt_path = find_ckpt(run_dir)
    if not ckpt_path:
        print(f"  SKIP {run_dir.name}: no checkpoint found")
        return None

    if config_path is None:
        config_path = find_config_for_run(run_dir)
    if config_path is None:
        print(f"  SKIP {run_dir.name}: no config found")
        return None

    print(f"  loading {ckpt_path.name} ...", end=" ", flush=True)
    model = load_model(config_path, ckpt_path, device)

    all_l1, all_lpips = [], []
    all_real_feats, all_fake_feats = [], []

    from torchmetrics.image.fid import FrechetInceptionDistance
    if not skip_rfid:
        fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
        fid_metric.eval()

    with torch.no_grad():
        for batch in val_loader:
            x = batch.to(device)
            xrec, _ = model(x)
            xrec = xrec.clamp(-1, 1)

            # Per-sample L1
            l1 = (x - xrec).abs().mean(dim=[1, 2, 3])
            all_l1.append(l1.cpu())

            # Per-sample LPIPS
            lp = lpips_fn(x, xrec)
            all_lpips.append(lp.cpu())

            if not skip_rfid:
                real_uint8 = ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8)
                fake_uint8 = ((xrec + 1) * 127.5).clamp(0, 255).to(torch.uint8)
                fid_metric.update(real_uint8, real=True)
                fid_metric.update(fake_uint8, real=False)

    all_l1 = torch.cat(all_l1).numpy()
    all_lpips = torch.cat(all_lpips).numpy()

    result = {
        "run": run_dir.name,
        "l1_mean": float(np.mean(all_l1)),
        "l1_std": float(np.std(all_l1)),
        "lpips_mean": float(np.mean(all_lpips)),
        "lpips_std": float(np.std(all_lpips)),
        "n_samples": len(all_l1),
    }

    if not skip_rfid:
        rfid_val = float(fid_metric.compute().item())
        result["rfid"] = rfid_val
        del fid_metric

    print(f"L1={result['l1_mean']:.4f}±{result['l1_std']:.4f}  "
          f"LPIPS={result['lpips_mean']:.4f}±{result['lpips_std']:.4f}"
          + (f"  rFID={result.get('rfid', 0):.2f}" if not skip_rfid else ""))

    del model
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--run_dirs_file", type=str, default=None,
                        help="text file with one run_dir per line")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--skip_rfid", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")

    # Build val loader
    split_dir = DATA_ROOT / "data" / "imagenet100_split_20k_5k"
    val_idx = np.load(split_dir / "val_idx.npy")
    train_files = sorted((DATA_ROOT / "data" / "imagenet100_raw" / "data").glob("train-*.parquet"))
    val_ds = ParquetImageDataset(train_files, size=128, train=False, indices=val_idx)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    print(f"val set: {len(val_ds)} images")

    # LPIPS
    lpips_fn = LPIPS().to(device).eval()

    # Collect run dirs
    run_dirs = []
    if args.run_dir:
        run_dirs.append(Path(args.run_dir))
    elif args.run_dirs_file:
        for line in open(args.run_dirs_file):
            line = line.strip()
            if line and not line.startswith("#"):
                run_dirs.append(Path(line))
    else:
        print("ERROR: specify --run_dir or --run_dirs_file")
        return

    config_path = Path(args.config) if args.config else None

    results = []
    for rd in run_dirs:
        print(f"\n=== {rd.name} ===")
        r = eval_run(rd, config_path, device, val_loader, lpips_fn,
                     skip_rfid=args.skip_rfid)
        if r:
            results.append(r)

    out_path = args.output or "val_std_results.jsonl"
    with open(out_path, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
