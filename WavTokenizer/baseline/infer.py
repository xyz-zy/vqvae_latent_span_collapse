"""
Docstring for WavTokenizer.infer

This script performs inference using a pretrained WavTokenizer model.
"""

# --coding:utf-8--
import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

# isort: off

# Local imports
from decoder.pretrained import WavTokenizer
from encoder.utils import convert_audio

# isort: on

"""
python infer2.py \
    --input_path ./test-other_filelist.txt \
    --config_path ../WavTokenizer_models/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml \
    --model_path ../WavTokenizer_models/WavTokenizer_small_320_24k_4096.ckpt \
    --out_name WavTokenizer_small_320_24k_4096

python infer2.py \
    --input_path ./test-clean_filelist.txt \
    --config_path ../WavTokenizer_models/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml \
    --model_path ../WavTokenizer_models/WavTokenizer_small_600_24k_4096.ckpt \
    --out_name WavTokenizer_small_600_24k_4096

python infer2.py \
    --config_path ./configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn_2gpu.yaml \
    --model_path ./result/train/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn_2gpu/lightning_logs/version_0/checkpoints/wavtokenizer_checkpoint_epoch=28_step=233160_val_loss=4.6529.ckpt \
    --out_name WavTokenizer_small_320_24k_4096_repl_2gpu_v0_28ep

python infer2.py \
    --input_path ./test-other_filelist.txt \
    --config_path ./configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn_8gpu_slurm.yaml \
    --model_path "./result/train/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn_8gpu_slurm/lightning_logs/version_1/checkpoints/wavtokenizer_checkpoint_epoch=58_step=118590_val_loss=4.4702.ckpt" \
    --out_name WavTokenizer_small_320_24k_4096_repl_8gpu_slurm_v1_58ep

python infer2.py \
    --input_path ./test_filelist.txt \
    --config_path ./configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn_8gpu_slurm.yaml \
    --model_path "./result/train/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn_8gpu_slurm/lightning_logs/version_2/checkpoints/wavtokenizer_checkpoint_epoch=0_step=2010_val_loss=8.8113.ckpt" \
    --out_name WavTokenizer_small_320_24k_4096_repl_8gpu_slurm_v1_0ep
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path", type=str, required=False, default=None, help="Path to model config"
)
parser.add_argument(
    "--model_path", type=str, required=True, help="Path to model checkpoint"
)
parser.add_argument(
    "--out_name",
    type=str,
    default="WavTokenizer_small_600_24k_4096",
    help="Subfolder name for output",
)
parser.add_argument(
    "--input_path",
    type=Path,
    default="./test-clean_filelist.txt",
    help="Path to input file list",
)
args = parser.parse_args()

model_path = Path(args.model_path)

config_path = (
    Path(args.config_path)
    if args.config_path
    else model_path.parent.parent / "config.yaml"
)
if not config_path.exists():
    raise FileNotFoundError(f"Config file not found at {config_path}")
config_path = str(config_path)

ll = args.out_name

device1 = torch.device("cuda:0")

input_path = args.input_path
out_folder = Path("./result/infer")

tmptmp = out_folder / ll

os.system("rm -r %s" % (tmptmp))
os.system("mkdir -p %s" % (tmptmp))


def align_mse(orig, recon):
    # orig, recon: torch.Tensor channels x samples or 1D
    if orig.dim() > 1:
        orig = torch.mean(orig, dim=0)
    if recon.dim() > 1:
        recon = torch.mean(recon, dim=0)
    L = min(orig.shape[-1], recon.shape[-1])
    if L == 0:
        return None
    diff = orig[:L] - recon[:L]
    return float(torch.mean(diff**2).item())


wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
wavtokenizer = wavtokenizer.to(device1)

with open(input_path, "r") as fin:
    x = fin.readlines()

x = [i.strip() for i in x]

model_out_folder = out_folder / ll
for i in tqdm(range(len(x))):
    input_path = Path(x[i])
    file_out_folder = model_out_folder / input_path.parent.relative_to(
        "/path/to/libritts/LibriTTS"
    )
    file_out_folder.mkdir(parents=True, exist_ok=True)
    audio_path = file_out_folder / input_path.name

    if os.path.exists(audio_path):
        continue
    wav, sr = torchaudio.load(input_path)

    bandwidth_id = torch.tensor([0])
    wav = wav.to(device1)

    features, discrete_code = wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)

    bandwidth_id = torch.tensor([0])
    bandwidth_id = bandwidth_id.to(device1)

    audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)

    torchaudio.save(
        audio_path,
        audio_out.cpu(),
        sample_rate=24000,
        encoding="PCM_S",
        bits_per_sample=16,
    )
