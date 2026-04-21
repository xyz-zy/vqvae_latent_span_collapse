# --coding:utf-8--
"""
Combined inference + codebook evaluation for WavTokenizer.

Encodes and decodes each audio file once, then:
- Saves reconstructed audio to {result_dir}/infer/{out_name}/  (for use by metrics/infer2.py)
- Saves codebook report JSON to {result_dir}/eval/{out_name}/{input_stem}/

Usage:
    python eval.py \
        --input_path test-clean_filelist.txt \
        --config_path configs/wavtokenizer_smalldata_frame75_3s_nq1_novq_dim512_kmeans200_attn_2gpu.yaml \
        --model_path /path/to/checkpoint.ckpt \
        --out_name novq_ep42
"""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchaudio
from decoder.pretrained import WavTokenizer
from tqdm import tqdm

LIBRITTS_ROOT = "/path/to/libritts/LibriTTS"


# ---------------------------------------------------------------------------
# Helpers (from codebook_eval.py)
# ---------------------------------------------------------------------------

def safe_flatten_codes(code_tensor):
    if isinstance(code_tensor, (list, tuple)):
        return np.concatenate([safe_flatten_codes(c) for c in code_tensor])
    if isinstance(code_tensor, torch.Tensor):
        arr = code_tensor.detach().cpu().numpy()
    else:
        arr = np.array(code_tensor)
    return arr.reshape(-1)


def mono(wave):
    if wave.dim() == 1:
        return wave
    return torch.mean(wave, dim=0)


def align_mse(orig, recon):
    orig = mono(orig)
    recon = mono(recon)
    L = min(orig.shape[-1], recon.shape[-1])
    if L == 0:
        return None
    return float(torch.mean((orig[:L] - recon[:L]) ** 2).item())


def compute_entropy_from_counts(counts):
    total = float(np.sum(counts))
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def get_vocab_size(config_path):
    try:
        with open(config_path, "r") as cf:
            cfg_text = cf.read()
        m = re.search(r"vq_bins\s*:\s*(\d+)", cfg_text)
        if not m:
            m = re.search(r"vocab_size\s*:\s*(\d+)", cfg_text)
        if not m:
            m = re.search(r"vq_bins\s*:\s*\[([^\]]+)\]", cfg_text)
        if m:
            grp = m.group(1)
            if "," in grp:
                first = re.search(r"(\d+)", grp)
                return int(first.group(1)) if first else None
            return int(grp)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, type=Path)
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--config_path", required=False, type=Path, default=None)
    parser.add_argument("--out_name", type=str, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample_rate", type=int, default=24000)
    parser.add_argument("--vocab_size", type=int, default=None,
                        help="Override vocab size from config (e.g. 65536)")
    parser.add_argument("--result_dir", type=Path, default=Path("./result"),
                        help="Root directory for infer/ and eval/ outputs")
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu"
    )
    print("Using device:", device)

    out_name = args.out_name or args.model_path.stem
    config_path = args.config_path if args.config_path else args.model_path.parent.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    infer_out = args.result_dir / "infer" / out_name
    eval_out = args.result_dir / "eval" / out_name / args.input_path.stem
    infer_out.mkdir(parents=True, exist_ok=True)
    eval_out.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {config_path}, {args.model_path}")
    wavtokenizer = WavTokenizer.from_pretrained0802(config_path, args.model_path)
    wavtokenizer = wavtokenizer.to(device)

    is_novq = getattr(wavtokenizer.feature_extractor, "novq", False)
    if is_novq:
        print("novq=True: skipping discrete codebook metrics, computing MSE only.")

    vocab_size = args.vocab_size if args.vocab_size is not None else get_vocab_size(config_path)
    if vocab_size is None and not is_novq:
        raise RuntimeError("vocab_size not found in config")
    if vocab_size is not None:
        print(f"vocab_size={vocab_size}")

    with open(args.input_path) as f:
        files = [l.strip() for l in f if l.strip()]

    all_codes = []
    per_file_unique = []
    per_file_len = []
    recon_mses = []
    quant_losses = []

    for p in tqdm(files):
        input_path = Path(p)
        try:
            wav, sr = torchaudio.load(input_path)
        except Exception as e:
            print(f"Failed to load {p}: {e}")
            continue

        # Determine output audio path (mirrors infer2.py layout)
        try:
            rel = input_path.parent.relative_to(LIBRITTS_ROOT)
        except ValueError:
            rel = Path(*input_path.parts[-3:-1])  # fallback: last 2 dirs
        file_out_dir = infer_out / rel
        file_out_dir.mkdir(parents=True, exist_ok=True)
        audio_out_path = file_out_dir / input_path.name

        wav = wav.to(device)
        bandwidth_id = torch.tensor([0], device=device)

        try:
            features, discrete_code = wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
        except Exception as e:
            print(f"encode_infer failed for {p}: {e}")
            continue

        try:
            audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
        except Exception as e:
            print(f"decode failed for {p}: {e}")
            continue

        # Save reconstructed audio
        if not audio_out_path.exists():
            torchaudio.save(
                audio_out_path,
                audio_out.cpu(),
                sample_rate=args.sample_rate,
                encoding="PCM_S",
                bits_per_sample=16,
            )

        # Collect MSE
        mse = align_mse(
            wav.squeeze(0) if wav.dim() > 1 else wav,
            audio_out.squeeze(0) if audio_out.dim() > 1 else audio_out,
        )
        if mse is not None:
            recon_mses.append(mse)

        # Collect discrete codes
        if discrete_code is not None:
            codes = safe_flatten_codes(discrete_code)
            if codes.size > 0:
                all_codes.append(codes)
                per_file_unique.append(len(np.unique(codes)))
                per_file_len.append(codes.size)

        # Quantization loss (non-novq only)
        if not is_novq:
            try:
                audio_in = wav.unsqueeze(1) if wav.dim() == 2 else wav
                emb = wavtokenizer.feature_extractor.encodec.encoder(audio_in)
                bw_list = wavtokenizer.feature_extractor.bandwidths
                bw_idx = int(bandwidth_id.view(-1).cpu().numpy()[0])
                bw_val = bw_list[bw_idx]
                q_res = wavtokenizer.feature_extractor.encodec.quantizer.infer(
                    emb, wavtokenizer.feature_extractor.frame_rate, bandwidth=bw_val
                )
                quant_losses.append(float(torch.mean((emb - q_res.quantized) ** 2).cpu().item()))
            except Exception:
                pass

    # Compute codebook report
    if all_codes:
        all_codes_np = np.concatenate(all_codes).astype(np.int64)
        counts = np.bincount(all_codes_np, minlength=int(vocab_size))
        used = int(np.sum(counts > 0))
        entropy = compute_entropy_from_counts(counts)
        report = {
            "total_files": len(files),
            "files_with_codes": len(per_file_len),
            "total_codes": int(all_codes_np.size),
            "vocab_size": int(vocab_size),
            "used_codes": used,
            "utilization": float(used) / float(vocab_size),
            "entropy_bits": entropy,
            "perplexity": float(2 ** entropy),
            "avg_unique_per_utterance": float(np.mean(per_file_unique)),
            "avg_seq_len": float(np.mean(per_file_len)),
            "recon_mse_mean": float(np.mean(recon_mses)) if recon_mses else None,
            "recon_mse_std": float(np.std(recon_mses)) if recon_mses else None,
            "quant_loss_mean": float(np.mean(quant_losses)) if quant_losses else None,
            "quant_loss_std": float(np.std(quant_losses)) if quant_losses else None,
        }
    elif is_novq:
        report = {
            "total_files": len(files),
            "files_with_codes": 0,
            "total_codes": None,
            "vocab_size": int(vocab_size) if vocab_size else None,
            "used_codes": None,
            "utilization": None,
            "entropy_bits": None,
            "perplexity": None,
            "avg_unique_per_utterance": None,
            "avg_seq_len": None,
            "recon_mse_mean": float(np.mean(recon_mses)) if recon_mses else None,
            "recon_mse_std": float(np.std(recon_mses)) if recon_mses else None,
            "quant_loss_mean": 0.0,
            "quant_loss_std": 0.0,
        }
    else:
        print("No codes collected; skipping codebook report.")
        return

    report_path = eval_out / f"codebook_report_{args.input_path.stem}.json"
    with open(report_path, "w") as fo:
        json.dump(report, fo, indent=2)
    print(f"Codebook report: {report_path}")
    print(f"Reconstructed audio: {infer_out}/")


if __name__ == "__main__":
    main()
