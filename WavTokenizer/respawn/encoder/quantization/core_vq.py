# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This implementation is inspired from
# https://github.com/lucidrains/vector-quantize-pytorch
# which is released under MIT License. Hereafter, the original license:
# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Core vector quantization implementation."""

import typing as tp
import warnings

from einops import rearrange, repeat
import torch
import torch.distributed as distributed
from torch import nn
import torch.nn.functional as F

from .. import distrib


def default(val: tp.Any, d: tp.Any) -> tp.Any:
    return val if val is not None else d


def ema_inplace(moving_avg, new, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]

def pad_shape(shape, size, dim = 0):
    return [size if i == dim else s for i, s in enumerate(shape)]

def sample_multinomial(total_count, probs):
    device = probs.device
    probs = probs.cpu()

    total_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype = torch.long)

    num_probs = len(probs)

    for i, prob in enumerate(probs):
        is_last = i == (num_probs - 1)

        s = torch.binomial(total_count, prob / remainder) if not is_last else total_count
        sample[i] = s
        total_count -= s
        remainder -= prob

    assert total_count == 0, f'invalid total count {total_count}'

    return sample.to(device)

def all_gather_sizes(x, dim):
    size = torch.tensor(x.shape[dim], dtype = torch.long, device = x.device)
    all_sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
    distributed.all_gather(all_sizes, size)
    return torch.stack(all_sizes)

def all_gather_variably_sized(x, sizes, dim = 0):
    rank = distributed.get_rank()
    all_x = []

    for i, size in enumerate(sizes):
        t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
        distributed.broadcast(t, src = i, async_op = True)
        all_x.append(t)

    distributed.barrier()
    return all_x

def sample_vectors_distributed(local_samples, num):
    local_samples = rearrange(local_samples, '1 ... -> ...')

    rank = distributed.get_rank()
    all_num_samples = all_gather_sizes(local_samples, dim = 0)

    if rank == 0:
        samples_per_rank = sample_multinomial(num, all_num_samples / all_num_samples.sum())
    else:
        samples_per_rank = torch.empty_like(all_num_samples)

    distributed.broadcast(samples_per_rank, src = 0)
    samples_per_rank = samples_per_rank.tolist()

    local_samples = sample_vectors(local_samples, samples_per_rank[rank])
    all_samples = all_gather_variably_sized(local_samples, samples_per_rank, dim = 0)
    out = torch.cat(all_samples, dim = 0)

    return rearrange(out, '... -> 1 ...')


def kmeans(
    samples,
    num_clusters: int,
    num_iters: int = 10,
    tol: float = 1e-4,
    return_history: bool = False,
):
    dim, dtype = samples.shape[-1], samples.dtype

    means = sample_vectors(samples, num_clusters)

    history = []

    for _ in range(num_iters):
        diffs = rearrange(samples, "n d -> n () d") - rearrange(
            means, "c d -> () c d"
        )
        dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        inertia = (-dists.gather(1, buckets[:, None])).sum().item()
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        new_means = torch.where(zero_mask[..., None], means, new_means)

        deltas = (new_means - means).norm(dim=-1)
        moved = (deltas > tol).sum().item()
        history.append(
            {
                "mean_delta": deltas.mean().item(),
                "max_delta": deltas.max().item(),
                "moved_centers": moved,
                "inertia": inertia,
            }
        )

        means = new_means

    if return_history:
        return means, bins, history
    return means, bins


def batched_bincount(x, *, minlength):
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype = dtype, device = device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target



def noop(*args, **kwargs):
    pass

def l2norm(t, dim = -1,  eps = 1e-6):
    return F.normalize(t, p = 2, dim = dim, eps = eps)

def batched_sample_vectors(samples, num):
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim = 0)], dim = 0)

from einops import reduce

def cdist(x, y, eps = 1e-8):
    x2 = reduce(x ** 2, 'b n d -> b n', 'sum')
    y2 = reduce(y ** 2, 'b n d -> b n', 'sum')
    xy = torch.einsum('b i d, b j d -> b i j', x, y) * -2
    return (rearrange(x2, 'b i -> b i 1') + rearrange(y2, 'b j -> b 1 j') + xy).clamp(min = eps).sqrt()


def kmeans_distrib(
    samples,
    num_clusters,
    num_iters = 10,
    use_cosine_sim = False,
    sample_fn = batched_sample_vectors,
    all_reduce_fn = noop,
    return_history = False,
    tol: float = 1e-4,
):
    num_codebooks, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device

    means = sample_fn(samples, num_clusters)
    history = [] if return_history else None

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ rearrange(means, 'h n d -> h d n')
        else:
            dists = -cdist(samples, means)

        buckets = torch.argmax(dists, dim = -1)
        if return_history:
            sel = dists.gather(-1, buckets.unsqueeze(-1)).squeeze(-1)
            inertia = (-sel).sum().float()
            all_reduce_fn(inertia)
        bins = batched_bincount(buckets, minlength = num_clusters)
        all_reduce_fn(bins)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype = dtype)

        new_means.scatter_add_(1, repeat(buckets, 'h n -> h n d', d = dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, '... -> ... 1')
        all_reduce_fn(new_means)

        if use_cosine_sim:
            new_means = l2norm(new_means)

        new_means = torch.where(
            rearrange(zero_mask, '... -> ... 1'),
            means,
            new_means
        )
        if return_history:
            deltas = (new_means - means).norm(dim = -1)
            moved = (deltas > tol).sum().item()
            history.append(
                {
                    "mean_delta": deltas.mean().item(),
                    "max_delta": deltas.max().item(),
                    "moved_centers": moved,
                    "inertia": inertia.item(),
                }
            )

        means = new_means
    if return_history:
        return means, bins, history
    return means, bins

class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.
    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch and use
            the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: int = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
        vq_accumulate_steps: int = 1,
    ):
        super().__init__()
        self.decay = decay
        init_fn: tp.Union[tp.Callable[..., torch.Tensor], tp.Any] = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.reset_cluster_size = threshold_ema_dead_code

        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("expired_codes_mask", torch.zeros(codebook_size, dtype=torch.bool), persistent=False)
        self.expired_codes = -1

        # Multi-step accumulation for codebook EMA updates
        self.vq_accumulate_steps = vq_accumulate_steps
        self.register_buffer("_accum_cluster_size", torch.zeros(codebook_size), persistent=False)
        self.register_buffer("_accum_embed_sum", torch.zeros(codebook_size, dim), persistent=False)
        self._accum_step_count = 0
        self.accum_nonzero = 0

        self.kmeans_history = None

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            return

        if distrib.is_distributed():
            data = rearrange(data, 'n d -> 1 n d')
            embed, cluster_size, kmeans_history = kmeans_distrib(
                data, self.codebook_size, self.kmeans_iters,
                sample_fn=sample_vectors_distributed,
                all_reduce_fn=distrib.all_reduce,
                return_history=True,
            )
            embed = embed[0]
            cluster_size = cluster_size[0]
        else:
            embed, cluster_size, kmeans_history = kmeans(data, self.codebook_size, self.kmeans_iters, return_history=True) #data不变
        self.kmeans_history = kmeans_history
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        # Make sure all buffers across workers are in sync after initialization
        distrib.broadcast_tensors(self.buffers())

    def replace_(self, samples, mask):
        if distrib.is_distributed():
            sampled = sample_vectors_distributed(rearrange(samples, '... -> 1 ...'), self.codebook_size)
            sampled = rearrange(sampled, '1 ... -> ...')
        else:
            sampled = sample_vectors(samples, self.codebook_size)
        modified_codebook = torch.where(
            mask[..., None], sampled, self.embed
        )
        self.embed.data.copy_(modified_codebook)

        reset_cluster_size = torch.full_like(self.cluster_size, self.reset_cluster_size * distrib.world_size())
        modified_cluster_size = torch.where(mask, reset_cluster_size, self.cluster_size)
        self.cluster_size.data.copy_(modified_cluster_size)

        reset_embed_avg = sampled * self.reset_cluster_size * distrib.world_size()
        modified_embed_avg = torch.where(mask[..., None], reset_embed_avg, self.embed_avg)
        self.embed_avg.data.copy_(modified_embed_avg)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            self.expired_codes = 0
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code * distrib.world_size()
        self.expired_codes = expired_codes.sum().item()
        self.expired_codes_mask = expired_codes
        if not torch.any(expired_codes):
            # self.expired_codes = 0
            return
        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        # distrib.broadcast_tensors(self.buffers())

    def preprocess(self, x):
        x = rearrange(x, "... d -> (...) d")
        return x

    def quantize(self, x):
        embed = self.embed.t()
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind):
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x):
        shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x)
        self._last_input = x.detach()

        self.init_embed_(x)

        embed_ind = self.quantize(x)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        quantize = self.dequantize(embed_ind)

        if self.training:
            # Stored for logging purposes
            self.embed_onehot_sum = embed_onehot.sum(0)

            # Only accumulate during generator steps (grad enabled),
            # not during discriminator steps (torch.no_grad).
            # Use no_grad for the accumulation itself to avoid building
            # autograd graph for EMA statistics.
            if torch.is_grad_enabled():
                with torch.no_grad():
                    cluster_size = embed_onehot.sum(0)
                    distrib.all_reduce(cluster_size)

                    embed_sum = x.t() @ embed_onehot
                    distrib.all_reduce(embed_sum)

                    self._accum_cluster_size.add_(cluster_size)
                    self._accum_embed_sum.add_(embed_sum.t())
                self._accum_step_count += 1

            # Only update codebook every N steps
            if self._accum_step_count >= self.vq_accumulate_steps:
                self.accum_nonzero = (self._accum_cluster_size > 0).sum().item()
                # print(f"[VQ UPDATE] accum_steps={self._accum_step_count}, "
                #       f"accum_cluster_size_sum={self._accum_cluster_size.sum().item():.1f}, "
                #       f"accum_nonzero={self.accum_nonzero}/{self.codebook_size}, "
                #       f"grad_enabled={torch.is_grad_enabled()}")
                ema_inplace(self.cluster_size, self._accum_cluster_size, self.decay)
                ema_inplace(self.embed_avg, self._accum_embed_sum, self.decay)

                cluster_size = (
                    laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
                    * self.cluster_size.sum()
                )
                embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
                self.embed.data.copy_(embed_normalized)

                # Expire dead codes — buffers are in sync so all workers
                # will take the same decision.
                self.expire_codes_(x)

                # Reset accumulators
                self._accum_cluster_size.zero_()
                self._accum_embed_sum.zero_()
                self._accum_step_count = 0

            # Sanity check that all the workers have the same codebook after the
            # update, to avoid silent errors.
            # if distrib.is_distributed():
            #     rank = distributed.get_rank()
            #     embed = self.embed.detach()

            #     if rank == 0:
            #         ref = embed.clone()
            #     else:
            #         ref = torch.empty_like(embed)

            #     distributed.broadcast(ref, src=0)
            #     max_diff = (embed - ref).abs().max()

            #     distributed.all_reduce(max_diff, op=distributed.ReduceOp.MAX)
            #     if rank == 0:
            #         print(f"[codebook sync] max_diff={max_diff.item():.6e}")


        return quantize, embed_ind


class VectorQuantization(nn.Module):
    """Vector quantization implementation.
    Currently supports only euclidean distance.
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
        commitment_weight (float): Weight for commitment loss.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        commitment_weight: float = 1.,
        vq_accumulate_steps: int = 1,
    ):
        super().__init__()
        _codebook_dim: int = default(codebook_dim, dim)

        requires_projection = _codebook_dim != dim
        self.project_in = (nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity())
        self.project_out = (nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity())

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight

        self._codebook = EuclideanCodebook(dim=_codebook_dim, codebook_size=codebook_size,
                                           kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
                                           decay=decay, epsilon=epsilon,
                                           threshold_ema_dead_code=threshold_ema_dead_code,
                                           vq_accumulate_steps=vq_accumulate_steps)
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x):
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def forward(self, x):

        # breakpoint()
        device = x.device
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        quantize, embed_ind = self._codebook(x)
        if self.training:
            quantize = x + (quantize - x).detach()
        loss = torch.tensor([0.0], device=device, requires_grad=self.training)

        if self.training:
            # warnings.warn('When using RVQ in training model, first check '
            #               'https://github.com/facebookresearch/encodec/issues/25 . '
            #               'The bug wasn\'t fixed here for reproducibility.')
            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight

        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize, embed_ind, loss


class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )

    def forward(self, x, n_q: tp.Optional[int] = None):
        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []

        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            quantized, indices, loss = layer(residual)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized
            all_indices.append(indices)
            all_losses.append(loss)

        out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
        return quantized_out, out_indices, out_losses

    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            all_indices.append(indices)
            quantized = layer.decode(indices)
            residual = residual - quantized.detach()
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out


class LanguageVectorQuantization(nn.Module):
    """Residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )
        # print("core_vq.py:self.layers",self.layers)

    def forward(self, x, n_q: tp.Optional[int] = None):
        # breakpoint()  x[b,t,c] #[64,75,128]  
        quantized_out = 0.0
        residual = x


        all_losses = []
        all_indices = []

        # breakpoint()

        n_q = n_q or len(self.layers)
          
        for layer in self.layers[:n_q]:
            quantized_out, indices, loss = layer(residual)  #得到该层的表征，该层的indices,该层的loss  [64,75]
            # residual = residual - quantized.detach()
            # quantized_out = quantized_out + quantized
            all_indices.append(indices)
            all_losses.append(loss)
        # breakpoint()
        # breakpoint()

        out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
        return quantized_out, out_indices, out_losses

    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            all_indices.append(indices)
            quantized = layer.decode(indices)
            residual = residual - quantized.detach()
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out
