import torch
import torch.nn as nn

from compressai.models import SimpleVAECompressionModel
from compressai.models.utils import conv, deconv
from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import (
    CheckerboardLatentCodec,
    GaussianConditionalLatentCodec,
    HyperLatentCodec,
    HyperpriorLatentCodec,
)

from compressai.layers import (
    CheckerboardMaskedConv2d,
    GDN,
)
from compressai.ops import quantize_ste
# from .GaussianMixture import GaussianMixtureConditionalLatentCodec
# from .Checkerboard import Checkerboard

class qlc(SimpleVAECompressionModel):
    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(**kwargs)
        
        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )
        self.h_a = nn.Sequential(
            conv(M, N, stride=2, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=1, kernel_size=1),
        )

        self.h_s = nn.Sequential(
            deconv(N, N, stride=2, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            deconv(N, N, stride=2, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=1),
        )
        
        self.e_a = nn.Sequential(            
            conv(M, M, stride=2, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(M, M, stride=2,kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(M, M, stride=1, kernel_size=1),
        )
        self.e_s = nn.Sequential(            
            deconv(M, M, stride=2, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            deconv(M, M, stride=2,kernel_size=3),
            nn.LeakyReLU(inplace=True),
            deconv(M, M, stride=1, kernel_size=1),
        )


        self.latent_codec = HyperpriorLatentCodec(
            latent_codec={
                "y": CheckerboardLatentCodec(
                    latent_codec={
                        "y": GaussianConditionalLatentCodec(quantizer="ste"),
                    },
                    entropy_parameters=nn.Sequential(
                        nn.Conv2d(M * 3, 640, 1),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(640, 640, 1),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(640, 2 * M, 1),
                    ),
                    context_prediction=CheckerboardMaskedConv2d(
                        M, 2 * M, kernel_size=5, stride=1, padding=2
                    ),
                ),
                "hyper": HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=self.h_a,
                    h_s=self.h_s,
                    quantizer="ste",
                ),
            },
        )

        self.residual_entropy = EntropyBottleneck(M)

    def forward(self, x):
        y = self.g_a(x)
        y_out = self.latent_codec(y)
        y_hat = y_out["y_hat"]

        residual = y - y_hat
        c = self.e_a(residual)
        _, c_likelihoods = self.residual_entropy(c)
        c_hat = quantize_ste(c)
        residual_hat =  torch.sigmoid(self.e_s(c_hat)) - 0.5

        y_out["likelihoods"]["c"] = c_likelihoods
        x_hat = self.g_s(y_hat+residual_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": y_out["likelihoods"],
        }

    def compress(self, x):
        y = self.g_a(x)
        outputs = self.latent_codec.compress(y)

        residual = y - outputs["y_hat"]
        c = self.e_a(residual)
        c_strings = self.residual_entropy.compress(c)

        outputs["strings"].extend(c_strings)
        outputs["shape"]["c"] = c.size()
        return outputs

    def decompress(self, strings, shape, **kwargs):
        y_out = self.latent_codec.decompress(strings, shape,**kwargs)
        c_hat = self.residual_entropy.decompress(strings[-1],size=shape["c"])
        residual_hat = torch.sigmoid(self.e_s(c_hat)) - 0.5
        y_hat = y_out["y_hat"]
        x_hat = self.g_s(y_hat+residual_hat).clamp_(0, 1)
        return {
            "x_hat": x_hat,
        }

