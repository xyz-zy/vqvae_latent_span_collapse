"""
Random-Sample Codebook Initialization for NoVQ WavTokenizer

1. Loads the best model checkpoint (epoch 42, val_loss=1.7049) trained with novq=True
2. Samples a random subset of the training dataset and extracts encoder latents
3. Subsamples N_LATENTS_FOR_KMEANS latent vectors
4. Randomly draws CODEBOOK_SIZE latents (without replacement) as the codebook
5. Saves the model with the random codebook
"""

# =============================================================================
# 1. Setup
# =============================================================================

import sys
import os

REPO_DIR = './'
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

import random
import json as _json
import numpy as np
import torch
import torchaudio
import soundfile
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for terminal use
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')

os.system('nvidia-smi')

# =============================================================================
# 2. Paths and Configuration
# =============================================================================

LOG_ROOT = (
    './result/train/'
    'wavtokenizer_smalldata_frame75_3s_nq1_novq_dim512_kmeans200_attn_2gpu/'
    'lightning_logs/version_0'
)
CONFIG_PATH    = os.path.join(LOG_ROOT, 'config.yaml')
CKPT_PATH      = os.path.join(
    LOG_ROOT, 'checkpoints',
    'wavtokenizer_checkpoint_epoch=42_step=345720_val_loss=1.7049.ckpt'
)
TRAIN_FILELIST = os.path.join(REPO_DIR, 'train_filelist.txt')
SAVE_DIR       = os.path.join(REPO_DIR, 'novq')
os.makedirs(SAVE_DIR, exist_ok=True)

BEST_EPOCH     = 42
SAMPLE_RATE    = 24_000
NUM_SAMPLES    = 72_000   # 3-second clips
LATENT_DIM     = 512
CODEBOOK_SIZE  = 65536
N_AUDIO_FILES  = 6000     # number of training files to sample
N_LATENTS_FOR_KMEANS = CODEBOOK_SIZE * 20  # 20× oversampling
CLUSTER_SIZE_TO_SAVE = 0.1

SAVE_DIR = os.path.join(SAVE_DIR, f'ep{BEST_EPOCH}_k{CODEBOOK_SIZE}_n{N_LATENTS_FOR_KMEANS}')
os.makedirs(SAVE_DIR, exist_ok=True)

print(f'Config:    {CONFIG_PATH}')
print(f'Checkpoint:{CKPT_PATH}')
print(f'Save dir:  {SAVE_DIR}')

# =============================================================================
# 3. Load Model Checkpoint
# =============================================================================

from decoder.pretrained import WavTokenizer

model = WavTokenizer.from_pretrained0802(CONFIG_PATH, CKPT_PATH)
model = model.to(DEVICE)
model.eval()

encoder = model.feature_extractor.encodec.encoder
print('Model loaded successfully.')
print(f'Encoder: {sum(p.numel() for p in encoder.parameters()):,} parameters')

# =============================================================================
# 4. Load Random Training Sample and Extract Latents
# =============================================================================

with open(TRAIN_FILELIST) as f:
    all_files = f.read().splitlines()

sampled_files = random.sample(all_files, min(N_AUDIO_FILES, len(all_files)))
print(f'Sampled {len(sampled_files)} audio files from {len(all_files)} total.')


def load_audio(path: str, target_sr: int = SAMPLE_RATE, num_samples: int = NUM_SAMPLES) -> torch.Tensor:
    """Load, resample, and crop/pad an audio file to a fixed length."""
    y, sr = soundfile.read(path)
    y = torch.tensor(y, dtype=torch.float32)
    if y.dim() > 1:
        y = y.mean(dim=-1)
    y = y.unsqueeze(0)  # (1, T)
    if sr != target_sr:
        y = torchaudio.functional.resample(y, sr, target_sr)
    if y.size(-1) < num_samples:
        repeat_y = y.repeat(1, 1 + num_samples // y.size(-1))
        y = torch.cat([y, repeat_y[:, :num_samples - y.size(-1)]], dim=1)
    else:
        start = random.randint(0, y.size(-1) - num_samples)
        y = y[:, start:start + num_samples]
    return y[0]  # (T,)


@torch.no_grad()
def extract_latents(model, audio_paths, batch_size=8, device=DEVICE):
    """Run encoder on a list of audio files and collect all latent vectors.

    Returns: (N, D) tensor of latent vectors, where N = n_files * n_frames.
    """
    encoder = model.feature_extractor.encodec.encoder
    all_latents = []

    for i in tqdm(range(0, len(audio_paths), batch_size), desc='Extracting latents'):
        batch_paths = audio_paths[i:i + batch_size]
        wavs = []
        for p in batch_paths:
            try:
                wavs.append(load_audio(p))
            except Exception as e:
                print(f'  Skipping {p}: {e}')
        if not wavs:
            continue

        wav_batch = torch.stack(wavs).unsqueeze(1).to(device)  # (B, 1, T)
        emb = encoder(wav_batch)                               # (B, D, frames)
        B, D, F = emb.shape
        latents_batch = emb.permute(0, 2, 1).reshape(-1, D).cpu()  # (B*F, D)
        all_latents.append(latents_batch)

    return torch.cat(all_latents, dim=0)  # (N, D)


latents = extract_latents(model, sampled_files)
latents = latents[torch.randperm(latents.size(0))[:N_LATENTS_FOR_KMEANS]]
print(f'Extracted latents: {latents.shape}  (N={latents.shape[0]}, D={latents.shape[1]})')
print(f'  mean={latents.mean():.4f}, std={latents.std():.4f}')

os.system('nvidia-smi')

# =============================================================================
# 5. Helpers
# =============================================================================

@torch.no_grad()
def compute_distortion(X: torch.Tensor, centroids: torch.Tensor, chunk: int = 4096) -> float:
    """Mean squared quantization error: mean_x min_c ||x - c||^2."""
    X = X.float()
    centroids = centroids.float()
    total = 0.0
    for start in range(0, len(X), chunk):
        x_chunk = X[start:start + chunk]
        d2 = (
            x_chunk.pow(2).sum(1, keepdim=True)
            - 2.0 * x_chunk @ centroids.t()
            + centroids.pow(2).sum(1, keepdim=True).t()
        )
        total += d2.clamp(min=0).min(dim=1).values.sum().item()
    return total / len(X)


def _assign_counts(centroids: torch.Tensor, X: torch.Tensor, chunk: int = 4096) -> np.ndarray:
    """Assign each latent to its nearest centroid; return per-centroid count array."""
    k = centroids.shape[0]
    counts = torch.zeros(k, dtype=torch.float32)
    C = centroids.float()
    for s in range(0, len(X), chunk):
        xc = X[s:s + chunk].float()
        d2 = (xc.pow(2).sum(1, keepdim=True)
              - 2.0 * xc @ C.t()
              + C.pow(2).sum(1, keepdim=True).t()).clamp(min=0)
        counts.scatter_add_(0, d2.argmin(dim=1), torch.ones(len(xc)))
    return counts.numpy()


from sklearn.decomposition import PCA

print('Fitting PCA on latents...')
_pca = PCA(n_components=2, random_state=42)
latents_2d = _pca.fit_transform(latents.float().numpy())
print(f'  Explained variance: PC1={_pca.explained_variance_ratio_[0]:.3f}, '
      f'PC2={_pca.explained_variance_ratio_[1]:.3f}')


def plot_pca_with_centers(centroids: torch.Tensor, title: str, save_name: str = None):
    """Project latents and centroids to the shared 2D PCA basis and scatter-plot them."""
    c_2d = _pca.transform(centroids.float().cpu().numpy())

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(latents_2d[:, 0], latents_2d[:, 1],
               s=2, alpha=0.15, color='steelblue',
               label=f'Latents (N={len(latents_2d):,})', rasterized=True)
    ax.scatter(c_2d[:, 0], c_2d[:, 1],
               s=8, alpha=0.6, color='crimson', marker='x', linewidths=0.6,
               label=f'Centers (K={len(c_2d):,})')
    ax.set_xlabel(f'PC1 ({_pca.explained_variance_ratio_[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({_pca.explained_variance_ratio_[1]:.1%} var)')
    ax.set_title(title)
    ax.legend(loc='upper right', markerscale=3)
    plt.tight_layout()
    if save_name:
        path = os.path.join(SAVE_DIR, save_name)
        plt.savefig(path, dpi=150)
        print(f'  Saved plot → {path}')
    plt.close(fig)


def plot_codebook_usage(centroids: torch.Tensor, title: str, save_name: str = None):
    """Compute and plot codebook usage statistics for a given set of centroids."""
    counts = _assign_counts(centroids.cpu(), latents.cpu())
    used = int((counts > 0).sum())
    utilization = used / len(counts)
    p = counts / counts.sum()
    p_pos = p[p > 0]
    entropy = float(-np.sum(p_pos * np.log2(p_pos)))
    perplexity = 2 ** entropy

    print(f'{title}:')
    print(f'  Used codes:  {used:,} / {len(counts):,}  ({utilization:.1%})')
    print(f'  Entropy:     {entropy:.3f} bits   Perplexity: {perplexity:.1f}')

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(counts, bins=50, log=True, color='steelblue', edgecolor='white', linewidth=0.3)
    ax.axvline(counts.mean(), color='crimson', linestyle='--', label=f'mean = {counts.mean():.1f}')
    ax.set_xlabel('Latents assigned to centroid')
    ax.set_ylabel('# centroids (log scale)')
    ax.set_title(f'{title} — codebook usage')
    ax.legend()
    plt.tight_layout()
    if save_name:
        path = os.path.join(SAVE_DIR, save_name)
        plt.savefig(path, dpi=150)
        print(f'  Saved plot → {path}')
    plt.close(fig)

# =============================================================================
# 6. Random-Sample Codebook
# =============================================================================

idx_rand = torch.randperm(len(latents))[:CODEBOOK_SIZE]
centroids_rand = latents[idx_rand].cpu()  # (K, D)
distortion_rand = compute_distortion(latents.cpu(), centroids_rand)
print(f'Random sample: distortion = {distortion_rand:.6f}')

plot_pca_with_centers(
    centroids_rand,
    'Random sample from latents',
    save_name=f'pca_rand_{N_LATENTS_FOR_KMEANS}.png',
)
plot_codebook_usage(
    centroids_rand,
    'Random sample from latents',
    save_name=f'usage_rand_{N_LATENTS_FOR_KMEANS}.png',
)

# =============================================================================
# 7. Save Model
# =============================================================================

def save_model_with_codebook(
    ckpt_path: str,
    save_path: str,
    centroids: torch.Tensor,
):
    """Load checkpoint, patch the codebook, and save to save_path."""
    raw = torch.load(ckpt_path, map_location="cpu")
    sd  = raw['state_dict']

    prefix = 'feature_extractor.encodec.quantizer.vq.layers.0._codebook'
    cluster_size = torch.full((centroids.shape[0],), CLUSTER_SIZE_TO_SAVE)

    sd[f'{prefix}.embed']        = centroids.float()
    sd[f'{prefix}.embed_avg']    = (centroids.float() * cluster_size.unsqueeze(1))
    sd[f'{prefix}.cluster_size'] = cluster_size
    sd[f'{prefix}.inited']       = torch.tensor([True])

    raw['state_dict'] = sd
    torch.save(raw, save_path)
    print(f'  Saved → {save_path}')


out_path = os.path.join(SAVE_DIR, f'ep{BEST_EPOCH}_rand_{N_LATENTS_FOR_KMEANS}_thresh{CLUSTER_SIZE_TO_SAVE}.ckpt')
print('Saving random-sample checkpoint...')
save_model_with_codebook(CKPT_PATH, out_path, centroids_rand.to("cpu"))
print('Checkpoint saved.')

# =============================================================================
# 8. Summary
# =============================================================================

counts = _assign_counts(centroids_rand.cpu(), latents.cpu())
used   = int((counts > 0).sum())
util   = used / len(counts)
p      = counts / counts.sum(); p_pos = p[p > 0]
ent    = float(-np.sum(p_pos * np.log2(p_pos)))

comparison = {
    'n_latents_for_kmeans': int(N_LATENTS_FOR_KMEANS),
    'methods': {
        'rand': {'distortion': distortion_rand, 'utilization': util, 'entropy_bits': ent}
    }
}

cmp_path = os.path.join(SAVE_DIR, f'comparison_{N_LATENTS_FOR_KMEANS}.json')
with open(cmp_path, 'w') as _f:
    _json.dump(comparison, _f, indent=2)
print(f'Saved comparison table → {cmp_path}')

print('=' * 65)
print('SUMMARY')
print('=' * 65)
print(f'Checkpoint used : epoch {BEST_EPOCH}, val_loss=1.7049')
print(f'Latent vectors  : {latents.shape[0]:,} (from {N_AUDIO_FILES} audio files)')
print(f'Codebook size   : {CODEBOOK_SIZE}')
print(f'Latent dim      : {LATENT_DIM}')
print()
print(f'Method          : Random sample from latents')
print(f'Distortion (MSE): {distortion_rand:.6f}')
print(f'Utilization     : {util:.1%}  ({used:,} / {CODEBOOK_SIZE:,} codes used)')
print(f'Entropy         : {ent:.3f} bits')
print(f'Saved as        : {out_path}')
print('=' * 65)
