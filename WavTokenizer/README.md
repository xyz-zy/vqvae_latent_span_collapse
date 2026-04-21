# WavTokenizer experiments

Three variants of the WavTokenizer fork; Each subdirectory is a self-contained working tree of the same upstream repo, ported from different branches used for the paper.

| Subdir      | Ported from                    | Source branch           | Role |
|-------------|--------------------------------|-------------------------|------|
| `baseline/` | `../WavTokenizer-clean`        | `clean`                 | Baseline WavTokenizer with added logging. |
| `respawn/`  | `../WavTokenizer-emafix`       | `emafix-fixrespawn`     | Baseline + corrected dead-codeword respawning (distributed k-means, cluster-size reset on respawn, matched-EMA update). |
| `novq/`     | `../WavTokenizer-novq`         | `novq-log-latent-space` | Baseline trained as an autoencoder (VQ disabled via `novq: true`), plus latent-space logging. |

## Pretrained eval weights

`wav2vec_small.pt` and `epoch=3-step=7459.ckpt` are stored once in `shared_metrics/` and symlinked into each subdir's `metrics/` dir. They will be downloaded the first time you run.

## Environment setup

All three subdirs target the same environment (Python 3.9 + CUDA). Conda is recommended — each subdir ships a `requirements.txt` with pinned versions.

```bash
conda create -n wavtokenizer python=3.9 -y
conda activate wavtokenizer

# PyTorch (match your CUDA); the pinned torch 2.0.0 can be built with cu118:
pip install torch==2.0.0 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# Everything else — all three subdirs share the same requirements
pip install -r baseline/requirements.txt
```

Sanity-check:

```bash
python -c "import torch, pytorch_lightning as pl; print(torch.__version__, pl.__version__, torch.cuda.is_available())"
```

The same env is used for training, the `quantize_novq_*.py` codebook-init scripts, and `eval.py`.

## Data

The training configs load audio via file lists at each subdir's root (`train_filelist.txt`, `dev_filelist.txt`, etc.) containing one absolute wav path per line. These lists are environment-specific and are not tracked by git — generate them once for your local LibriTTS copy using the helper script that lives in each subdir:

```bash
cd baseline     # same script lives in respawn/ and novq/

# Dev split (combine dev-clean + dev-other into dev_filelist.txt)
python generate_file_list.py \
  --input_dir /path/to/LibriTTS/dev-clean /path/to/LibriTTS/dev-other \
  --output_file dev_filelist.txt

# Train split (e.g. train-clean-100 + train-clean-360)
python generate_file_list.py \
  --input_dir /path/to/LibriTTS/train-clean-100 /path/to/LibriTTS/train-clean-360 \
  --output_file train_filelist.txt

# Subsampled train for quick runs — 10% of train-clean-100
python generate_file_list.py \
  --input_dir /path/to/LibriTTS/train-clean-100 \
  --downsample_frac 0.1 \
  --output_file train_0.1_filelist.txt
```

`generate_file_list.py --help` lists all options; sampling uses a fixed seed (`42`) so the same fraction gives the same files across runs. If you want matching filelists across all three subdirs, generate once and symlink or copy.

## Test run

```bash

# baseline smoke test
cd baseline && python train.py fit \
  --config configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn_2gpu_test.yaml

# respawn (emafix) smoke test — same config name
cd ../respawn && python train.py fit \
  --config configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn_2gpu_test.yaml

# novq smoke test
cd ../novq && python train.py fit \
  --config configs/wavtokenizer_smalldata_frame75_3s_nq1_novq_dim512_kmeans200_attn_2gpu_test.yaml
```

Each test config requests 2 GPUs and a 2M step schedule — override with `--trainer.devices=1` or `--trainer.max_steps=N` for short runs.

## Run baselines

Full end-to-end VQ baselines live under `baseline/configs/` — one config per codebook size (k=4096/8192/16384) and framerate (40 / 75 fps). Run them directly:

```bash
cd baseline

# Single-run example
python train.py fit \
  --config configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn_2gpu.yaml

# Or sweep all 2-GPU configs sequentially
for cfg in configs/wavtokenizer_smalldata_*_attn_2gpu.yaml; do
  python train.py fit --config "$cfg"
done
```

`*_test.yaml` variants exist for quick smoke checks; the non-test configs are the real 2M-step runs. Outputs land in `baseline/result/train/<run-name>/`.

## Autoencoder Warm-up Followed by VQ Training

The "AE → VQ" pipeline trains an autoencoder first (no quantization), then initializes a codebook offline from the AE's latents, then fine-tunes with the fixed-respawn VQ trainer.

**1. Train the no-VQ autoencoder** (from `novq/`):

```bash
cd novq
python train.py fit \
  --config configs/wavtokenizer_smalldata_frame75_3s_nq1_novq_dim512_kmeans200_attn_2gpu.yaml
```

This drops a checkpoint at `novq/result/train/<run-name>/lightning_logs/version_0/checkpoints/wavtokenizer_checkpoint_epoch=42_step=345720_val_loss=1.7049.ckpt`. The `quantize_novq_*.py` scripts are hard-coded to look for exactly that path — if you want to quantize a different checkpoint, edit `CKPT_PATH` at the top of the script.

**2. Initialize a codebook from the AE latents** — pick the script matching your target codebook size:

| Script | K | Output filename |
|---|---|---|
| `quantize_novq_10x_4096.py` | 4096 | `novq/ep42_k4096_n40960/ep42_kmeans_sklearnpp_40960_thresh1.0.ckpt` |
| `quantize_novq_10x_8192.py` | 8192 | `novq/ep42_k8192_n81920/ep42_kmeans_sklearnpp_81920_thresh0.5.ckpt` |
| `quantize_novq_10x_16384.py` | 16384 | `novq/ep42_k16384_n163840/ep42_kmeans_sklearnpp_163840_thresh0.25.ckpt` |
| `quantize_novq_20x_rand_65536.py` | 65536 (random, no k-means) | `novq/ep42_k65536_n1310720/ep42_rand_1310720_thresh0.1.ckpt` |

```bash
# still in novq/
python quantize_novq_10x_8192.py
```

Each script extracts encoder latents from random training clips, runs sklearn k-means++ (or random sampling for the 65k script), patches the codebook into the checkpoint's `state_dict`, and writes a new `.ckpt` plus PCA/usage diagnostic plots under the save dir.

**3. Resume from the quantized checkpoint with the fixed-respawn trainer** (`respawn/`):

```bash
cd ../respawn
python train.py fit \
  --config configs/wavtokenizer_smalldata_frame75_3s_nq1_novq_dim512_kmeans200_attn_2gpu_code8192_sklearnpp_81920_thresh0.25.yaml
```

Each respawn config's `resume_model:` already points at the corresponding quantize-script output via `../novq/novq/ep{EPOCH}_k{K}_n{N}/...`. The pairing is:

| respawn config | resumes from |
|---|---|
| `...code4096_sklearnpp_40960_thresh0.5.yaml` | `quantize_novq_10x_4096.py` output |
| `...code8192_sklearnpp_81920_thresh0.25.yaml` | `quantize_novq_10x_8192.py` output |
| `...code16384_sklearnpp_163840_thresh0.125.yaml` | `quantize_novq_10x_16384.py` output |
| `...code65536_rand_thresh0.03.yaml` | `quantize_novq_20x_rand_65536.py` output |
