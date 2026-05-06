"""Plain autoencoder variant of VQModel: encoder -> quant_conv -> post_quant_conv -> decoder.

Keeps the 1x1 quant/post_quant convs so the bottleneck width is still controlled by `embed_dim`
(matches VQModel architecture for apples-to-apples comparison), but drops the vector quantizer
entirely. The `VQLPIPSWithDiscriminator` loss still runs; the codebook term receives zero.
"""
import torch

from taming.models.vqgan import VQModel


class AEModel(VQModel):
    def __init__(self, ddconfig, lossconfig, embed_dim, n_embed=1, **kwargs):
        # Initialize VQModel normally (which builds a VectorQuantizer), then
        # replace the quantizer with Identity so it contributes no params.
        super().__init__(ddconfig=ddconfig, lossconfig=lossconfig,
                         n_embed=n_embed, embed_dim=embed_dim, **kwargs)
        self.quantize = torch.nn.Identity()

    def encode(self, x):
        h = self.encoder(x)
        z = self.quant_conv(h)
        return z

    def decode(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        xrec = self.decode(z)
        qloss = torch.zeros((), device=x.device)
        return xrec, qloss
