from typing import Any, Dict, List, Optional, Tuple, Union

import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import GaussianMixtureConditional
from compressai.ops import quantize_ste

from compressai.latent_codecs import LatentCodec



class GaussianMixtureConditionalLatentCodec(LatentCodec):
    gaussian_conditional: GaussianMixtureConditional
    entropy_parameters: nn.Module

    def __init__(
        self,
        K=3,
        scale_table: Optional[Union[List, Tuple]] = None,
        gaussian_conditional: Optional[GaussianMixtureConditional] = None,
        entropy_parameters: Optional[nn.Module] = None,
        quantizer: str = "noise",
        chunks: Tuple[str] = ("scales", "means", "weights"),
        **kwargs,
    ):
        super().__init__()
        self.quantizer = quantizer
        self.gaussian_conditional = gaussian_conditional or GaussianMixtureConditional(
            K=K, scale_table=scale_table, **kwargs
        )
        self.entropy_parameters = entropy_parameters or nn.Identity()
        self.chunks = tuple(chunks)

    def forward(self, y: Tensor, ctx_params: Tensor) -> Dict[str, Any]:
        gaussian_params = self.entropy_parameters(ctx_params)
        scales_hat, means_hat, weights_hat = self._chunk(gaussian_params)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales=scales_hat, means=means_hat, weights=weights_hat)
        if self.quantizer == "ste":
            y_hat = quantize_ste(y - means_hat) + means_hat
        return {"likelihoods": {"y": y_likelihoods}, "y_hat": y_hat}

    def compress(self, y: Tensor, ctx_params: Tensor) -> Dict[str, Any]:
        gaussian_params = self.entropy_parameters(ctx_params)
        scales_hat, means_hat, weights_hat = self._chunk(gaussian_params)
        ans, _ = self.gaussian_conditional.compress(y, scales_hat, means_hat, weights_hat)
        y_strings = ans[0]
        abs_max = ans[1]
        zero_bitmap = ans[2]
        y_hat = self.gaussian_conditional.decompress(
            y_strings, abs_max, zero_bitmap,scales=scales_hat, means=means_hat, weights=weights_hat
        )
        return {"strings": [y_strings], "shape": y.shape[2:4], "y_hat": y_hat, "abs_max": abs_max, "zero_bitmap": zero_bitmap}

    def decompress(
        self,
        strings: List[List[bytes]],
        shape: Tuple[int, int],
        ctx_params: Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        (y_strings,) = strings
        gaussian_params = self.entropy_parameters(ctx_params)
        scales_hat, means_hat, weights_hat = self._chunk(gaussian_params)
        y_hat = self.gaussian_conditional.decompress(
            y_strings, kwargs["abs_max"], kwargs["zero_bitmap"],scales=scales_hat, means=means_hat, weights=weights_hat
        )
        assert y_hat.shape[2:4] == shape
        return {"y_hat": y_hat}

    def _chunk(self, params: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # scales, means, weights = None, None, None
        if self.chunks == ("scales", "means", "weights"):
            scales, means, weights = params.chunk(3, 1)
        if self.chunks == ("means", "scales", "weights"):
            means, scales, weights = params.chunk(3, 1)
        return scales, means, weights
