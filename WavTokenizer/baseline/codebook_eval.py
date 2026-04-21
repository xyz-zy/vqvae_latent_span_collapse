# --coding:utf-8--
"""
Evaluate codebook usage over a dataset using a pretrained WavTokenizer.

Produces a JSON report with:
- total_codes: total number of discrete codes observed
- vocab_size: inferred or reported codebook size
- used_codes: number of distinct codes used
- utilization: fraction used_codes / vocab_size
- entropy: Shannon entropy (bits)
- perplexity: 2**entropy
- avg_unique_per_utterance: mean number of distinct codes per file
- avg_seq_len: mean number of tokens per file
- avg_recon_mse: average reconstruction MSE between original and decoded audio (if decode succeeds)

Usage:
python codebook_eval.py --input_path path/to/list.txt --config_path ... --model_path ...
"""

import argparse
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchaudio
from decoder.pretrained import WavTokenizer
from tqdm import tqdm


def safe_flatten_codes(code_tensor):
    # Accept many shapes: (1, L), (B, L), list/tuple, numpy
    if isinstance(code_tensor, (list, tuple)):
        code_tensor = np.concatenate([safe_flatten_codes(c) for c in code_tensor])
        return code_tensor
    if isinstance(code_tensor, torch.Tensor):
        arr = code_tensor.detach().cpu().numpy()
    else:
        arr = np.array(code_tensor)
    return arr.reshape(-1)


def mono(wave):
    # wave: Tensor (channels, samples)
    if wave.dim() == 1:
        return wave
    return torch.mean(wave, dim=0)


def align_mse(orig, recon):
    # compute MSE between two 1D tensors; align to min length
    orig = mono(orig)
    recon = mono(recon)
    L = min(orig.shape[-1], recon.shape[-1])
    if L == 0:
        return None
    diff = orig[:L] - recon[:L]
    return float(torch.mean(diff**2).item())


def compute_entropy_from_counts(counts):
    total = float(np.sum(counts))
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def get_vocab_size(config_path):
    # if not found on model, try to parse config yaml for common keys (e.g. vq_bins)
    vocab_size = None
    try:
        with open(config_path, "r") as cf:
            cfg_text = cf.read()
        # look for patterns like 'vq_bins: 4096' or 'vocab_size: 4096'
        m = re.search(r"vq_bins\s*:\s*(\d+)", cfg_text)
        if not m:
            m = re.search(r"vocab_size\s*:\s*(\d+)", cfg_text)
        if not m:
            m = re.search(r"vq_bins\s*:\s*\[([^\]]+)\]", cfg_text)
        if m:
            # if matched list, take first int; else parse int
            grp = m.group(1)
            if "," in grp:
                first = re.search(r"(\d+)", grp)
                vocab_size = int(first.group(1)) if first else None
            else:
                vocab_size = int(grp)
            print(f"Inferred vocab size from config ({config_path}):", vocab_size)
    except Exception:
        vocab_size = None
    return vocab_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        required=True,
        type=Path,
        help="File listing dataset audio paths, one per line",
    )
    parser.add_argument(
        "--out_folder",
        default="./result/eval",
        type=Path,
        help="Output folder for report",
    )
    parser.add_argument("--config_path", required=False, type=Path, default=None, help="WavTokenizer config yaml")
    parser.add_argument("--model_path", required=True, type=Path, help="WavTokenizer checkpoint")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument(
        "--sample_rate", type=int, default=24000, help="Sample rate for saving/decoding"
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default=None,
        help="Model name override",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu"
    )
    print("Using device:", device)

    out_name = args.out_name or args.model_path.stem
    out_folder = args.out_folder / out_name / args.input_path.stem
    os.makedirs(out_folder, exist_ok=True)

    model_path = args.model_path
    config_path = args.config_path if args.config_path else model_path.parent.parent / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    print("Loading WavTokenizer from", config_path, model_path)
    wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
    wavtokenizer = wavtokenizer.to(device)

    # infer vocab size: prefer model attribute, then config file
    vocab_size = get_vocab_size(config_path)
    if vocab_size is None:
        raise RuntimeError(" vocab_size not found in config")

    with open(args.input_path, "r") as f:
        files = [l.strip() for l in f if l.strip()]

    all_codes = []
    per_file_unique = []
    per_file_len = []
    recon_mses = []
    quant_losses = []

    for i, p in tqdm(enumerate(files), total=len(files)):
        try:
            wav, sr = torchaudio.load(p)
        except Exception as e:
            print(f"Failed to load {p}: {e}")
            continue

        wav = wav.to(device)
        bandwidth_id = torch.tensor([0], device=device)

        # get codes/features via encode_infer (keeps existing behavior)
        try:
            features, discrete_code = wavtokenizer.encode_infer(
                wav, bandwidth_id=bandwidth_id
            )
        except Exception as e:
            print(f"encode_infer failed for {p}: {e}")
            continue

        codes = safe_flatten_codes(discrete_code)
        if codes.size == 0:
            continue

        all_codes.append(codes)
        unique_codes = np.unique(codes)
        per_file_unique.append(len(unique_codes))
        per_file_len.append(codes.size)

        # attempt decode and compute MSE
        try:
            audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
            if isinstance(audio_out, torch.Tensor):
                mse = align_mse(
                    wav.squeeze(0) if wav.dim() > 1 else wav,
                    audio_out.squeeze(0) if audio_out.dim() > 1 else audio_out,
                )
                if mse is not None:
                    recon_mses.append(mse)
        except Exception:
            pass

        # compute quantization loss: L2 between encoder latent and quantized (dequantized) vector
        try:
            # replicate encoder + quantizer inference to obtain encoder latent and quantized output
            # prepare audio in same shape as feature extractor expects
            audio_in = wav
            if audio_in.dim() == 2:
                audio_for_encoder = audio_in.unsqueeze(1)
            else:
                audio_for_encoder = audio_in

            emb = wavtokenizer.feature_extractor.encodec.encoder(audio_for_encoder)
            # select bandwidth value (float) from feature_extractor.bandwidths
            bw_list = wavtokenizer.feature_extractor.bandwidths
            bw_idx = int(bandwidth_id.view(-1).cpu().numpy()[0]) if isinstance(bandwidth_id, torch.Tensor) else int(bandwidth_id)
            bw_val = bw_list[bw_idx]
            q_res = wavtokenizer.feature_extractor.encodec.quantizer.infer(emb, wavtokenizer.feature_extractor.frame_rate, bandwidth=bw_val)
            # emb and q_res.quantized are both (B, D, L) so compute MSE
            qloss = float(torch.mean((emb - q_res.quantized) ** 2).cpu().item())
            quant_losses.append(qloss)
        except Exception:
            pass

    if len(all_codes) == 0:
        print("No codes collected; exiting.")
        return

    all_codes = np.concatenate(all_codes).astype(np.int64)
    total_codes = all_codes.size

    counts = np.bincount(all_codes, minlength=int(vocab_size))
    used = int(np.sum(counts > 0))
    utilization = float(used) / float(vocab_size) if vocab_size > 0 else -1.0
    entropy = compute_entropy_from_counts(counts)
    perplexity = float(2**entropy)

    avg_unique_per_utt = float(np.mean(per_file_unique)) if per_file_unique else 0.0
    avg_seq_len = float(np.mean(per_file_len)) if per_file_len else 0.0
    recon_mse_mean = float(np.mean(recon_mses)) if recon_mses else None
    recon_mse_std = float(np.std(recon_mses)) if recon_mses else None
    quant_loss_mean = float(np.mean(quant_losses)) if quant_losses else None
    quant_loss_std = float(np.std(quant_losses)) if quant_losses else None

    report = {
        "total_files": len(files),
        "files_with_codes": len(per_file_len),
        "total_codes": int(total_codes),
        "vocab_size": int(vocab_size),
        "used_codes": int(used),
        "utilization": utilization,
        "entropy_bits": entropy,
        "perplexity": perplexity,
        "avg_unique_per_utterance": avg_unique_per_utt,
        "avg_seq_len": avg_seq_len,
        "recon_mse_mean": recon_mse_mean,
        "recon_mse_std": recon_mse_std,
        "quant_loss_mean": quant_loss_mean,
        "quant_loss_std": quant_loss_std,
    }
    input_filestem= args.input_path.stem

    out_path = os.path.join(out_folder, f"codebook_report_{input_filestem}.json")
    with open(out_path, "w") as fo:
        json.dump(report, fo, indent=2)

    # also save counts for inspection
    # counts_path = os.path.join(out_folder, "code_counts.npy")
    # np.save(counts_path, counts)

    print("Wrote report to", out_path)


if __name__ == "__main__":
    main()
