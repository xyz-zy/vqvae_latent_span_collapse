"""Full step-0 evaluation with rFID for all fresh AE checkpoints × codebook sizes."""
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

AE_STEPS = [1000, 5000, 10000, 15000, 20000, 30000, 40000, 100000]
CODEBOOKS = {
    1024:  "configs/imagenet_vqgan_ref128_nodisc_cb1k_respawn.yaml",
    16384: "configs/imagenet_vqgan_ref128_nodisc_cb16k_respawn.yaml",
    65536: "configs/imagenet_vqgan_ref128_nodisc_cb65k_respawn.yaml",
}

def to_uint8(x):
    return ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8)

def main():
    device = torch.device("cuda:0")

    split_dir = DATA_ROOT / "data" / "imagenet100_split_20k_5k"
    val_idx = np.load(split_dir / "val_idx.npy")
    train_idx = np.load(split_dir / "train_idx.npy")
    val_files = sorted((DATA_ROOT / "data" / "imagenet100_raw" / "data").glob("validation-*.parquet"))
    train_files = sorted((DATA_ROOT / "data" / "imagenet100_raw" / "data").glob("train-*.parquet"))
    val_ds = ParquetImageDataset(val_files, size=128, train=False, indices=val_idx)
    train_ds = ParquetImageDataset(train_files, size=128, train=True, indices=train_idx)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
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
            print(f"skipping ae_steps={ae_steps}: {ckpt_path} not found")
            continue

        for n_embed, cfg_path in CODEBOOKS.items():
            print(f"\n=== AE {ae_steps}, cb={n_embed} ===", flush=True)
            cfg = OmegaConf.load(cfg_path)
            from taming.models.vqgan import VQModel
            mp = cfg.model.params
            model = VQModel(
                ddconfig=OmegaConf.to_container(mp.ddconfig),
                lossconfig=OmegaConf.to_container(mp.lossconfig),
                n_embed=n_embed, embed_dim=mp.embed_dim,
                vq_kwargs=OmegaConf.to_container(mp.get("vq_kwargs", {})) or {},
            ).to(device)

            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state = ckpt["model"] if "model" in ckpt else ckpt
            model.load_state_dict(state, strict=False)

            # k-means init
            model.eval()
            quantize = model.quantize
            samples = []
            loader_iter = iter(train_loader)
            n_batches = max(32, (n_embed + 1023) // 1024 + 1)
            for _ in range(n_batches):
                batch = next(loader_iter).to(device)
                with torch.no_grad():
                    h = model.quant_conv(model.encoder(batch))
                    h = h.permute(0, 2, 3, 1).reshape(-1, h.shape[1]).contiguous()
                samples.append(h)
            z_samples = torch.cat(samples, dim=0)
            quantize.kmeans_init_codebook(z_samples, n_iters=20, seed=42, n_batches=n_batches)
            del samples, z_samples
            torch.cuda.empty_cache()

            # Evaluate with rFID
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
                    n_img += x.shape[0]
                    n_batch += 1
                    _, _, info = model.encode(x)
                    _, _, idx = info
                    usage.scatter_add_(0, idx.view(-1).long(),
                                       torch.ones(idx.view(-1).shape[0], dtype=torch.long, device=device))
                    fid.update(to_uint8(xrec), real=False)

            rec_loss = rec_sum / n_batch
            lpips = lpips_sum / n_batch
            num_used = int((usage > 0).sum().item())
            total = float(usage.sum().item())
            p = usage.float() / max(total, 1.0)
            nz = p[p > 0]
            H_bits = float(-(nz * torch.log2(nz)).sum().item()) if nz.numel() else 0.0

            w = quantize.embedding.weight.data.float()
            w_c = w - w.mean(dim=0, keepdim=True)
            _, S, _ = torch.linalg.svd(w_c, full_matrices=False)
            var = S ** 2
            cumvar = var.cumsum(0) / var.sum()
            eff_dim = int((cumvar < 0.99).sum().item()) + 1

            rfid = float(fid.compute().item())

            row = {
                "ae_steps": ae_steps, "n_embed": n_embed,
                "rec_loss": round(rec_loss, 4), "l1": round(rec_loss - lpips, 4),
                "lpips": round(lpips, 4), "mse": round(mse_sum / n_img, 5),
                "codebook_used": num_used, "util_pct": round(num_used / n_embed * 100, 1),
                "entropy_bits": round(H_bits, 2), "eff_dim_99": eff_dim,
                "rfid": round(rfid, 2),
            }
            results.append(row)
            print(f"  rec={row['rec_loss']:.4f} l1={row['l1']:.4f} lpips={row['lpips']:.4f} "
                  f"used={num_used}/{n_embed} ({row['util_pct']}%) dim={eff_dim} rFID={rfid:.2f}",
                  flush=True)

            del model
            torch.cuda.empty_cache()

    out = ROOT / "ae_kmeans_step0_eval_fresh_full.jsonl"
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
