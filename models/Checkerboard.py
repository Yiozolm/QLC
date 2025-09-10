from compressai.latent_codecs import CheckerboardLatentCodec, LatentCodec
from typing import Any, Dict, List, Mapping, Optional, Tuple
from torch import Tensor


class Checkerboard(CheckerboardLatentCodec):
    def __init__(
        self,
        latent_codec: Optional[Mapping[str, LatentCodec]] = None,
        entropy_parameters: Optional[nn.Module] = None,
        context_prediction: Optional[nn.Module] = None,
        anchor_parity="even",
        forward_method="twopass",
        **kwargs,
    ):
        super().__init__(latent_codec=latent_codec, 
        entropy_parameters=entropy_parameters, 
        context_prediction=context_prediction, 
        anchor_parity=anchor_parity, 
        forward_method=forward_method,
        **kwargs)

    def compress(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        n, c, h, w = y.shape
        y_hat_ = side_params.new_zeros((2, n, c, h, w // 2))
        side_params_ = self.unembed(side_params)
        y_ = self.unembed(y)
        y_strings_ = [None] * 2
        abs_maxs = list()
        zero_bitmaps = list()

        for i in range(2):
            y_ctx_i = self.unembed(self.context_prediction(self.embed(y_hat_)))[i]
            if i == 0:
                y_ctx_i = self._mask(y_ctx_i, "all")
            params_i = self.entropy_parameters(self.merge(y_ctx_i, side_params_[i]))
            y_out = self.latent_codec["y"].compress(y_[i], params_i)
            y_hat_[i] = y_out["y_hat"]
            [y_strings_[i]] = y_out["strings"]
            abs_maxs[i] = y_out["abs_max"]
            zero_bitmaps[i] = y_out["zero_bitmap"]

        y_hat = self.embed(y_hat_)

        return {
            "strings": y_strings_,
            "shape": y_hat.shape[1:],
            "y_hat": y_hat,
            "abs_max": abs_maxs,
            "zero_bitmap": zero_bitmaps,
        }

    def decompress(
        self,
        strings: List[List[bytes]],
        shape: Tuple[int, ...],
        side_params: Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        y_strings_ = strings
        n = len(y_strings_[0])
        assert len(y_strings_) == 2
        assert all(len(x) == n for x in y_strings_)

        c, h, w = shape
        y_i_shape = (h, w // 2)
        y_hat_ = side_params.new_zeros((2, n, c, h, w // 2))
        side_params_ = self.unembed(side_params)

        for i in range(2):
            y_ctx_i = self.unembed(self.context_prediction(self.embed(y_hat_)))[i]
            if i == 0:
                y_ctx_i = self._mask(y_ctx_i, "all")
            params_i = self.entropy_parameters(self.merge(y_ctx_i, side_params_[i]))
            y_out = self.latent_codec["y"].decompress(
                [y_strings_[i]], y_i_shape, params_i, abs_max=kwargs['abs_max'][i], zero_bitmap=kwargs['zero_bitmap'][i], 
            )
            y_hat_[i] = y_out["y_hat"]

        y_hat = self.embed(y_hat_)

        return {
            "y_hat": y_hat,
        }