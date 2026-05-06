"""Plot AE latent variances with reverse water-filling levels.

Usage:
  python3 plot_ae_waterfilling.py --ae_step 40000              # full 1×2 plot
  python3 plot_ae_waterfilling.py --ae_step 40000 --pca_only   # PCA panel only
  python3 plot_ae_waterfilling.py --ae_step 40000 --ae_run_dir /path/to/ae_run
"""
import argparse
import sys
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from train_small2 import ParquetImageDataset

DATA_ROOT = Path(".")
ROOTS = [
    Path("results"),
]
OUT = ROOTS[0]

CB_CONFIGS = [
    (10, "$|\mathcal{C}|$=1k (R=10)", "tab:blue"),
    (14, "$|\mathcal{C}|$=16k (R=14)", "tab:orange"),
    (16, "$|\mathcal{C}|$=65k (R=16)", "tab:green"),
]


def find_ae_ckpt(ae_dir, step):
    matches = sorted(Path(ae_dir).glob(f"ckpts/step_{step:07d}_*.pt"))
    return matches[0] if matches else None


def find_ae_run(step):
    for root in ROOTS:
        for d in sorted(root.glob("*_ae_ref128_constlr_*"), reverse=True):
            if find_ae_ckpt(d, step):
                return d
    return None


def compute_waterfill(variances, R_total):
    variances = np.sort(variances)[::-1]
    lo, hi = 1e-20, np.max(variances) * 2
    for _ in range(200):
        theta = (lo + hi) / 2
        rate = sum(max(0, 0.5 * np.log2(v / theta)) for v in variances)
        if rate > R_total:
            lo = theta
        else:
            hi = theta
    theta = (lo + hi) / 2
    active = int(np.sum(variances > theta))
    return float(theta), active


def collect_latents(ae_dir, step, device):
    ckpt_path = find_ae_ckpt(ae_dir, step)
    if not ckpt_path:
        raise FileNotFoundError(f"No AE ckpt for step {step} in {ae_dir}/ckpts/")

    cfg_path = ae_dir / "config.yaml"
    if not cfg_path.exists():
        cfg_path = DATA_ROOT / "configs" / "imagenet_ae_ref128_nodisc.yaml"
    cfg = OmegaConf.load(cfg_path)
    mp = cfg.model.params

    from taming.models.ae import AEModel
    model = AEModel(
        ddconfig=OmegaConf.to_container(mp.ddconfig),
        lossconfig=OmegaConf.to_container(mp.lossconfig),
        embed_dim=mp.embed_dim,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    split_dir = DATA_ROOT / "data" / "imagenet100_split_20k_5k"
    train_idx = np.load(split_dir / "train_idx.npy")
    train_files = sorted((DATA_ROOT / "data" / "imagenet100_raw" / "data").glob("train-*.parquet"))
    train_ds = ParquetImageDataset(train_files, size=128, train=False, indices=train_idx)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    all_z = []
    with torch.no_grad():
        for batch in train_loader:
            x = batch.to(device)
            h = model.quant_conv(model.encoder(x))
            h = h.permute(0, 2, 3, 1).reshape(-1, h.shape[1]).contiguous()
            all_z.append(h.cpu())
    z = torch.cat(all_z, dim=0).float()
    print(f"collected {z.shape[0]} latent vectors, dim={z.shape[1]}")
    del model
    torch.cuda.empty_cache()
    return z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae_step", type=int, default=40000)
    parser.add_argument("--ae_run_dir", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--pca_only", action="store_true", help="plot only the PCA basis variances panel")
    args = parser.parse_args()

    ae_dir = Path(args.ae_run_dir) if args.ae_run_dir else find_ae_run(args.ae_step)
    if not ae_dir or not ae_dir.exists():
        print(f"ERROR: AE run dir not found"); return
    print(f"AE run: {ae_dir}")

    device = torch.device(f"cuda:{args.device}")
    z = collect_latents(ae_dir, args.ae_step, device)

    # PCA variances (eigenvalues)
    z_centered = z - z.mean(dim=0, keepdim=True)
    _, S, _ = torch.linalg.svd(z_centered, full_matrices=False)
    pca_var = (S ** 2 / (z.shape[0] - 1)).numpy()

    # Coordinate-wise variances (sorted descending) — skip if pca_only
    if not args.pca_only:
        coord_var = z.var(dim=0).numpy()
        coord_var_sorted = np.sort(coord_var)[::-1]

    # Water-filling for each codebook size
    wf_pca = {}
    wf_coord = {}
    for R, label, _ in CB_CONFIGS:
        theta_p, active_p = compute_waterfill(pca_var, R)
        wf_pca[R] = (theta_p, active_p)
        if not args.pca_only:
            theta_c, active_c = compute_waterfill(coord_var_sorted, R)
            wf_coord[R] = (theta_c, active_c)
            print(f"{label}: coord $D*$={theta_c:.2f} dims={active_c}  |  PCA $D*$={theta_p:.2f} dims={active_p}")
        else:
            print(f"{label}: PCA $D*$={theta_p:.2f} dims={active_p}")

    step_k = args.ae_step // 1000

    # Plot
    if args.pca_only:
        fig, ax2 = plt.subplots(1, 1, figsize=(5, 5))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Left: coordinate-wise variances
        ax1.bar(range(len(coord_var_sorted)), coord_var_sorted, width=1.0, color="gray",
                alpha=0.5, label="Coordinate variance")
        for R, label, color in CB_CONFIGS:
            theta, active = wf_coord[R]
            ax1.axhline(theta, color=color, ls="--", lw=1.5,
                         label=f"{label}: $D*$={theta:.2f}, {active} dims")
        # ax1.set_yscale("log")
        ax1.set_xlabel("Coordinate Index (sorted by variance)")
        ax1.set_ylabel("Variance")
        ax1.set_title("Coordinate-Wise Variances")
        ax1.legend(fontsize=10, loc="upper right")
        ax1.grid(True, alpha=0.3)

    # PCA variances
    ax2.bar(range(len(pca_var)), pca_var, width=1.0, color="gray",
            alpha=0.5, label="PCA eigenvalue")
    for R, label, color in CB_CONFIGS:
        theta, active = wf_pca[R]
        ax2.axhline(theta, color=color, ls="--", lw=1.5,
                     label=f"{label}: $D*$={theta:.2f}, {active} dims")
    # ax2.set_yscale("log")
    ax2.set_xlabel("Principal Component Index")
    ax2.set_ylabel("Variance (Eigenvalue)")
    if not args.pca_only:
        ax2.set_title("PCA Basis Variances")
    ax2.legend(fontsize=10, loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.suptitle(f"Reverse Water-Filling at AE {step_k}k Checkpoint\n$|\mathcal{{C}}|$ = 1k, 16k, 65k",
                 fontsize=13, y=1.05)

    suffix = "_pca" if args.pca_only else ""
    out = OUT / f"ae_{step_k}k_variances_waterfilling_all_cb{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
