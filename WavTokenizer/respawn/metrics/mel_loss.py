"""
Compute mel-spectrogram reconstruction loss from saved audio pairs.

Reads original audio paths from a filelist and matches them against
reconstructed audio already on disk (same layout as eval.py / infer2.py).

Usage:
    python metrics/mel_loss.py \
        --input_path test-clean_filelist.txt \
        --prepath /path/to/result/infer/out_name
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decoder.loss import MelSpecReconstructionLoss

LIBRITTS_ROOT = "/path/to/libritts/LibriTTS"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, type=Path,
                        help="Filelist of original audio paths")
    parser.add_argument("--prepath", required=True, type=Path,
                        help="Directory with reconstructed audio (from eval.py)")
    parser.add_argument("--sample_rate", type=int, default=24000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu"
    )
    mel_loss_fn = MelSpecReconstructionLoss(sample_rate=args.sample_rate).to(device)

    with open(args.input_path) as f:
        files = [l.strip() for l in f if l.strip()]

    mel_losses = []
    for p in files:
        input_path = Path(p)

        # Mirror eval.py layout to find reconstructed audio
        try:
            rel = input_path.parent.relative_to(LIBRITTS_ROOT)
        except ValueError:
            rel = Path(*input_path.parts[-3:-1])
        recon_path = args.prepath / rel / input_path.name

        if not recon_path.exists():
            print(f"Skipping {p}: reconstructed file not found at {recon_path}")
            continue

        try:
            wav, _ = torchaudio.load(input_path)
            recon, _ = torchaudio.load(recon_path)
        except Exception as e:
            print(f"Failed to load {p}: {e}")
            continue

        wav = wav.to(device)
        recon = recon.to(device)

        with torch.no_grad():
            L = min(wav.shape[-1], recon.shape[-1])
            ml = mel_loss_fn(recon[..., :L], wav[..., :L])
            mel_losses.append(float(ml.item()))

    if not mel_losses:
        print("No files processed.")
        return

    result = {
        "mel_loss_mean": float(np.mean(mel_losses)),
        "mel_loss_std": float(np.std(mel_losses)),
        "num_files": len(mel_losses),
    }

    out_path = args.prepath / "mel_loss.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"mel_loss_mean: {result['mel_loss_mean']:.4f}")
    print(f"mel_loss_std:  {result['mel_loss_std']:.4f}")
    print(f"num_files:     {result['num_files']}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
