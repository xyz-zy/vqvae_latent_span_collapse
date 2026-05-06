"""Compute water-filling for VQGAN encoder latents at each checkpoint."""
import torch, numpy as np, json, sys, argparse, glob
from pathlib import Path
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))
from train_small2 import ParquetImageDataset

DATA_ROOT = Path(".")

def compute_waterfill(variances, R_total):
    variances = np.sort(variances)[::-1]
    lo, hi = 1e-20, np.max(variances) * 2
    for _ in range(200):
        theta = (lo + hi) / 2
        rate = sum(max(0, 0.5 * np.log2(v / theta)) for v in variances)
        if rate > R_total: lo = theta
        else: hi = theta
    theta = (lo + hi) / 2
    active = int(np.sum(variances > theta))
    return round(float(theta), 4), active

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--steps", type=int, nargs="*", default=None)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "ckpts"

    split_dir = DATA_ROOT / "data" / "imagenet100_split_20k_5k"
    train_idx = np.load(split_dir / "train_idx.npy")
    train_files = sorted((DATA_ROOT / "data" / "imagenet100_raw" / "data").glob("train-*.parquet"))
    train_ds = ParquetImageDataset(train_files, size=128, train=False, indices=train_idx)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    rates = [(10, "cb1k"), (14, "cb16k"), (16, "cb65k")]

    if args.steps:
        ckpts = []
        for s in args.steps:
            matches = sorted(ckpt_dir.glob(f"step_{s:07d}_*.pt"))
            if matches: ckpts.append(matches[0])
    else:
        ckpts = sorted(ckpt_dir.glob("step_*.pt"))

    results = []
    for ckpt_path in ckpts:
        step = int(ckpt_path.name.split("_")[1])
        print(f"step {step}...", end=" ", flush=True)

        cfg = OmegaConf.load(args.config)
        from taming.models.vqgan import VQModel
        mp = cfg.model.params
        model = VQModel(
            ddconfig=OmegaConf.to_container(mp.ddconfig),
            lossconfig=OmegaConf.to_container(mp.lossconfig),
            n_embed=mp.n_embed, embed_dim=mp.embed_dim,
            vq_kwargs=OmegaConf.to_container(mp.get("vq_kwargs", {})) or {},
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state, strict=True)
        model.eval()

        all_z = []
        with torch.no_grad():
            for batch in train_loader:
                x = batch.to(device)
                h = model.quant_conv(model.encoder(x))
                h = h.permute(0, 2, 3, 1).reshape(-1, h.shape[1]).contiguous()
                all_z.append(h.cpu())
        z = torch.cat(all_z, dim=0).float()
        z_centered = z - z.mean(dim=0, keepdim=True)
        _, S, _ = torch.linalg.svd(z_centered, full_matrices=False)
        pca_var = (S ** 2 / (z.shape[0] - 1)).numpy()

        row = {"step": step}
        for R, label in rates:
            theta, active = compute_waterfill(pca_var, R)
            row[f"theta_R{R}"] = theta
            row[f"active_R{R}"] = active
            print(f"R={R}:θ={theta:.2f},d={active}", end="  ", flush=True)
        results.append(row)
        print()
        del model, all_z, z, z_centered, S; torch.cuda.empty_cache()

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.output}")

if __name__ == "__main__":
    main()
