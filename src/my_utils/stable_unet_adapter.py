from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    ImageHintTimeEmbedding,
    ImageProjection,
    ImageTimeEmbedding,
    PositionNet,
    TextImageProjection,
    TextImageTimeEmbedding,
    TextTimeEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unet_2d_blocks import (
    UNetMidBlock2D,
    UNetMidBlock2DCrossAttn,
    UNetMidBlock2DSimpleCrossAttn,
    get_down_block,
    get_up_block,
)

from einops import rearrange

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class UNet2DConditionOutput(BaseOutput):
    sample: torch.FloatTensor = None


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g + self.b


class ZeroConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(ZeroConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class MLP(nn.Module):
    def __init__(self, f_number, excitation_factor=2) -> None:
        super().__init__()
        self.norm = LayerNorm(f_number)
        self.pwconv1 = nn.Conv2d(f_number, int(excitation_factor * f_number), kernel_size=1)
        self.pwconv2 = nn.Conv2d(int(excitation_factor * f_number) // 2, f_number, kernel_size=1)

    def forward(self, x):
        x = self.norm(x)
        x = self.pwconv1(x)
        y1, y2 = x.chunk(2, dim=1)
        x = y1 * y2
        x = self.pwconv2(x)
        return x


class LowHighSplit(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.blur = nn.Conv2d(channels, channels, kernel_size, groups=channels, padding=padding, bias=False)
        with torch.no_grad():
            w = torch.zeros(channels, 1, kernel_size, kernel_size)
            w[:] = 1.0 / (kernel_size * kernel_size)
            self.blur.weight.copy_(w)

    def forward(self, x: torch.Tensor):
        lf = self.blur(x)
        hf = x - lf
        return lf, hf


class SE(nn.Module):
    def __init__(self, c: int, r: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, max(1, c // r), 1),
            nn.ReLU(True),
            nn.Conv2d(max(1, c // r), c, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        return x * self.fc(self.pool(x))


class FreqSplitAdapter(nn.Module):
    def __init__(self, width: int, prior_in_channels: int = 256, stage: str = "mid", learnable_alpha: bool = True):
        super().__init__()
        assert stage in {"down", "mid", "up"}
        self.stage = stage
        self.conv_in = nn.Conv2d(prior_in_channels, width, kernel_size=1)
        self.split = LowHighSplit(width)

        self.save_attention = False
        self.attn_cache = {}

        init_lf = 0.7 if stage == "down" else (0.5 if stage == "mid" else 0.3)
        init_hf = 0.3 if stage == "down" else (0.5 if stage == "mid" else 0.7)
        if learnable_alpha:
            self.alpha_lf = nn.Parameter(torch.tensor(init_lf, dtype=torch.float32))
            self.alpha_hf = nn.Parameter(torch.tensor(init_hf, dtype=torch.float32))
        else:
            self.register_buffer("alpha_lf", torch.tensor(init_lf, dtype=torch.float32))
            self.register_buffer("alpha_hf", torch.tensor(init_hf, dtype=torch.float32))

        self.get_kv = nn.Sequential(LayerNorm(width), nn.Conv2d(width, width * 2, 1))
        self.get_q = nn.Sequential(LayerNorm(width), nn.Conv2d(width, width, 1))
        self.out = ZeroConv2d(width, width)

    def forward(self, x: torch.Tensor, prior: torch.Tensor):
        prior = F.interpolate(self.conv_in(prior), size=x.shape[2:], mode="bilinear", align_corners=False)
        lf, hf = self.split(prior)
        prior_fused = self.alpha_lf * lf + self.alpha_hf * hf

        k, v = self.get_kv(prior_fused).chunk(2, dim=1)
        q = self.get_q(x)

        B, C, H, W = x.shape
        q_ = rearrange(q, "B C H W -> B C (H W)")
        k_ = rearrange(k, "B C H W -> B C (H W)")
        v_ = rearrange(v, "B C H W -> B C (H W)")

        attn_logits = (q_.transpose(-2, -1) @ k_) / (C ** 0.5)
        attn = attn_logits.softmax(dim=-1)

        out = attn @ v_.transpose(-2, -1)
        out = rearrange(out.transpose(-2, -1), "B C (H W) -> B C H W", H=H, W=W)

        residual = self.out(out)
        final = x + residual

        if self.save_attention:
            self.attn_cache = {
                "x_before_q": x.detach(),
                "low_freq": lf.detach(),
                "high_freq": hf.detach(),
                "pre_kv_feature": prior_fused.detach(),
                "q": q.detach(),
                "k": k.detach(),
                "v": v.detach(),
                "attn_logits": attn_logits.detach(),
                "attn_softmax": attn.detach(),
                "residual_branch": residual.detach(),
                "out_after_residual": final.detach(),
            }

        return final


# class FreqSplitAdapter(nn.Module):
#     def __init__(self, width: int, prior_in_channels: int = 256, stage: str = "mid", learnable_alpha: bool = True):
#         super().__init__()
#         assert stage in {"down", "mid", "up"}
#         self.stage = stage
#         self.conv_in = nn.Conv2d(prior_in_channels, width, kernel_size=1)
#         self.split = LowHighSplit(width)

#         init_lf = 0.7 if stage == "down" else (0.5 if stage == "mid" else 0.3)
#         init_hf = 0.3 if stage == "down" else (0.5 if stage == "mid" else 0.7)
#         if learnable_alpha:
#             self.alpha_lf = nn.Parameter(torch.tensor(init_lf, dtype=torch.float32))
#             self.alpha_hf = nn.Parameter(torch.tensor(init_hf, dtype=torch.float32))
#         else:
#             self.register_buffer("alpha_lf", torch.tensor(init_lf, dtype=torch.float32))
#             self.register_buffer("alpha_hf", torch.tensor(init_hf, dtype=torch.float32))

#         self.get_kv = nn.Sequential(LayerNorm(width), nn.Conv2d(width, width * 2, 1))
#         self.get_q = nn.Sequential(LayerNorm(width), nn.Conv2d(width, width, 1))
#         self.out = ZeroConv2d(width, width)

#     def forward(self, x: torch.Tensor, prior: torch.Tensor):
#         prior = F.interpolate(self.conv_in(prior), size=x.shape[2:], mode="bilinear", align_corners=False)
#         lf, hf = self.split(prior)
#         prior_fused = self.alpha_lf * lf + self.alpha_hf * hf

#         k, v = self.get_kv(prior_fused).chunk(2, dim=1)
#         q = self.get_q(x)

#         B, C, H, W = x.shape
#         q = rearrange(q, "B C H W -> B C (H W)")
#         k = rearrange(k, "B C H W -> B C (H W)")
#         v = rearrange(v, "B C H W -> B C (H W)")
#         attn = (q.transpose(-2, -1) @ k).softmax(dim=-1)
#         out = attn @ v.transpose(-2, -1)
#         out = rearrange(out.transpose(-2, -1), "B C (H W) -> B C H W", H=H, W=W)
#         return x + self.out(out)


# class PyramidAdapter(nn.Module):
#     def __init__(self, width: int, prior_in_channels: int = 256, stage: str = "mid"):
#         super().__init__()
#         assert stage in {"down", "mid", "up"}
#         self.stage = stage
#         self.proj = nn.Conv2d(prior_in_channels * 3, width, 1)
#         self.se = SE(width)
#         self.get_kv = nn.Sequential(LayerNorm(width), nn.Conv2d(width, width * 2, 1))
#         self.get_q = nn.Sequential(LayerNorm(width), nn.Conv2d(width, width, 1))
#         self.out = ZeroConv2d(width, width)

#         bias = {"down": (0.6, 0.3, 0.1), "mid": (0.33, 0.33, 0.34), "up": (0.2, 0.3, 0.5)}[stage]
#         self.register_buffer("w0", torch.tensor(bias[0], dtype=torch.float32))
#         self.register_buffer("w1", torch.tensor(bias[1], dtype=torch.float32))
#         self.register_buffer("w2", torch.tensor(bias[2], dtype=torch.float32))

#     def _pyr(self, p: torch.Tensor, size):
#         p0 = F.interpolate(p, size=size, mode="bilinear", align_corners=False)
#         p1 = F.interpolate(F.avg_pool2d(p, 2, 2), size=size, mode="bilinear", align_corners=False)
#         p2 = F.interpolate(F.avg_pool2d(p, 4, 4), size=size, mode="bilinear", align_corners=False)
#         return torch.cat([self.w0 * p0, self.w1 * p1, self.w2 * p2], dim=1)

#     def forward(self, x: torch.Tensor, prior: torch.Tensor):
#         size = x.shape[2:]
#         prior = self._pyr(prior, size)
#         prior = self.proj(prior)
#         prior = self.se(prior)

#         k, v = self.get_kv(prior).chunk(2, dim=1)
#         q = self.get_q(x)

#         B, C, H, W = x.shape
#         q = rearrange(q, "B C H W -> B C (H W)")
#         k = rearrange(k, "B C H W -> B C (H W)")
#         v = rearrange(v, "B C H W -> B C (H W)")
#         attn = (q.transpose(-2, -1) @ k).softmax(dim=-1)
#         out = attn @ v.transpose(-2, -1)
#         out = rearrange(out.transpose(-2, -1), "B C (H W) -> B C H W", H=H, W=W)
#         return x + self.out(out)

class PyramidAdapter(nn.Module):
    def __init__(self, width: int, prior_in_channels: int = 256, stage: str = "mid"):
        super().__init__()
        assert stage in {"down", "mid", "up"}
        self.stage = stage

        self.proj = nn.Conv2d(prior_in_channels * 3, width, 1)
        self.se = SE(width)
        self.get_kv = nn.Sequential(LayerNorm(width), nn.Conv2d(width, width * 2, 1))
        self.get_q = nn.Sequential(LayerNorm(width), nn.Conv2d(width, width, 1))
        self.out = ZeroConv2d(width, width)

        self.save_attention = False
        self.attn_cache = {}

        bias = {"down": (0.6, 0.3, 0.1), "mid": (0.33, 0.33, 0.34), "up": (0.2, 0.3, 0.5)}[stage]
        self.register_buffer("w0", torch.tensor(bias[0], dtype=torch.float32))
        self.register_buffer("w1", torch.tensor(bias[1], dtype=torch.float32))
        self.register_buffer("w2", torch.tensor(bias[2], dtype=torch.float32))

    def _pyr(self, p: torch.Tensor, size):
        p0 = F.interpolate(p, size=size, mode="bilinear", align_corners=False)
        p1 = F.interpolate(F.avg_pool2d(p, 2, 2), size=size, mode="bilinear", align_corners=False)
        p2 = F.interpolate(F.avg_pool2d(p, 4, 4), size=size, mode="bilinear", align_corners=False)
        return torch.cat([self.w0 * p0, self.w1 * p1, self.w2 * p2], dim=1)

    def forward(self, x: torch.Tensor, prior: torch.Tensor):
        size = x.shape[2:]

        # QKV 前融合特征
        prior = self._pyr(prior, size)
        prior = self.proj(prior)
        prior = self.se(prior)

        k, v = self.get_kv(prior).chunk(2, dim=1)
        q = self.get_q(x)

        B, C, H, W = x.shape

        q_ = rearrange(q, "B C H W -> B C (H W)")
        k_ = rearrange(k, "B C H W -> B C (H W)")
        v_ = rearrange(v, "B C H W -> B C (H W)")

        # QK^T / sqrt(dk)
        attn_logits = (q_.transpose(-2, -1) @ k_) / (C ** 0.5)

        # softmax attention
        attn = attn_logits.softmax(dim=-1)

        out = attn @ v_.transpose(-2, -1)
        out = rearrange(out.transpose(-2, -1), "B C (H W) -> B C H W", H=H, W=W)

        residual = self.out(out)
        final = x + residual

        if self.save_attention:
            with torch.no_grad():
                attn_f = attn.detach().float()
                logits_f = attn_logits.detach().float()

                Nq = attn_f.shape[1]
                Nk = attn_f.shape[2]
                uniform = 1.0 / float(Nk)

                # --------------------------------------------------
                # 1. mean attention: 参考用，通常偏平
                # --------------------------------------------------
                attn_mean = attn_f.mean(dim=1).view(B, 1, H, W)

                # --------------------------------------------------
                # 2. mean attention excess: 减掉 uniform baseline
                # --------------------------------------------------
                attn_mean_excess = (attn_f.mean(dim=1) - uniform).clamp_min(0.0)
                attn_mean_excess = attn_mean_excess.view(B, 1, H, W)

                # --------------------------------------------------
                # 3. max over query: 哪些 key 位置曾经被强关注
                # --------------------------------------------------
                attn_maxq = attn_f.amax(dim=1).view(B, 1, H, W)

                attn_maxq_excess = (attn_f.amax(dim=1) - uniform).clamp_min(0.0)
                attn_maxq_excess = attn_maxq_excess.view(B, 1, H, W)

                # --------------------------------------------------
                # 4. top-k query attention: 最推荐论文可视化
                # --------------------------------------------------
                top_ratio = 0.05
                k_top = max(1, int(Nq * top_ratio))

                query_strength = attn_f.amax(dim=-1)  # [B, Nq]
                top_idx = query_strength.topk(k_top, dim=1).indices

                attn_topk_list = []
                attn_topk_excess_list = []

                for b in range(B):
                    selected = attn_f[b, top_idx[b], :]  # [K, Nk]

                    topk_map = selected.mean(dim=0)
                    topk_excess_map = (topk_map - uniform).clamp_min(0.0)

                    attn_topk_list.append(topk_map.view(1, 1, H, W))
                    attn_topk_excess_list.append(topk_excess_map.view(1, 1, H, W))

                attn_topk = torch.cat(attn_topk_list, dim=0)
                attn_topk_excess = torch.cat(attn_topk_excess_list, dim=0)

                # --------------------------------------------------
                # 5. QK logits map: softmax 前原始相关性
                # --------------------------------------------------
                logits_z = (
                    logits_f - logits_f.mean(dim=-1, keepdim=True)
                ) / logits_f.std(dim=-1, keepdim=True).clamp_min(1e-6)

                qk_logits_pos = logits_z.clamp_min(0.0).mean(dim=1).view(B, 1, H, W)
                qk_logits_max = logits_z.clamp_min(0.0).amax(dim=1).view(B, 1, H, W)

                # --------------------------------------------------
                # 6. QKV 前融合特征能量
                # --------------------------------------------------
                prior_energy = prior.detach().float().pow(2).mean(dim=1, keepdim=True).sqrt()

                # --------------------------------------------------
                # 7. 最后残差影响
                # --------------------------------------------------
                residual_energy = residual.detach().float().pow(2).mean(dim=1, keepdim=True).sqrt()
                residual_absmax = residual.detach().float().abs().amax(dim=1, keepdim=True)

                self.attn_cache = {
                    # 主图最推荐
                    "attn_topk_excess": attn_topk_excess.detach().cpu(),

                    # 备选
                    "attn_maxq_excess": attn_maxq_excess.detach().cpu(),
                    "attn_mean_excess": attn_mean_excess.detach().cpu(),

                    # 参考
                    "attn_mean": attn_mean.detach().cpu(),
                    "attn_maxq": attn_maxq.detach().cpu(),
                    "attn_topk": attn_topk.detach().cpu(),

                    # QK^T / sqrt(dk)
                    "qk_logits_pos": qk_logits_pos.detach().cpu(),
                    "qk_logits_max": qk_logits_max.detach().cpu(),

                    # QKV 前融合输出
                    "prior_energy": prior_energy.detach().cpu(),

                    # 最后残差加号影响
                    "residual_energy": residual_energy.detach().cpu(),
                    "residual_absmax": residual_absmax.detach().cpu(),
                }

        return final




# class EdgeGateAdapter(nn.Module):
#     def __init__(self, width: int, prior_in_channels: int = 256, ksize: int = 3):
#         super().__init__()
#         self.conv_in = nn.Conv2d(prior_in_channels, width, 1)

#         self.sx = nn.Conv2d(1, 1, ksize, padding=ksize // 2, bias=False)
#         self.sy = nn.Conv2d(1, 1, ksize, padding=ksize // 2, bias=False)
#         with torch.no_grad():
#             # 目前默认 3x3 sobel（ksize=3 时严格匹配）
#             sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
#             sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
#             if self.sx.weight.shape == sx.shape:
#                 self.sx.weight.copy_(sx)
#                 self.sy.weight.copy_(sy)

#         self.get_kv = nn.Sequential(LayerNorm(width), nn.Conv2d(width, width * 2, 1))
#         self.get_q = nn.Sequential(LayerNorm(width), nn.Conv2d(width, width, 1))
#         self.out = ZeroConv2d(width, width)

#     def _edge_gate(self, x: torch.Tensor):
#         gimg = x.mean(1, keepdim=True)
#         ex, ey = self.sx(gimg), self.sy(gimg)
#         mag = torch.sqrt(ex**2 + ey**2)
#         gate = torch.sigmoid(mag.mean(dim=[1, 2, 3], keepdim=True))  # [B,1,1,1]
#         return gate

#     def forward(self, x: torch.Tensor, prior: torch.Tensor):
#         prior = F.interpolate(self.conv_in(prior), size=x.shape[2:], mode="bilinear", align_corners=False)
#         k, v = self.get_kv(prior).chunk(2, dim=1)
#         q = self.get_q(x)

#         B, C, H, W = x.shape
#         q = rearrange(q, "B C H W -> B C (H W)")
#         k = rearrange(k, "B C H W -> B C (H W)")
#         v = rearrange(v, "B C H W -> B C (H W)")
#         attn = (q.transpose(-2, -1) @ k).softmax(dim=-1)
#         out = attn @ v.transpose(-2, -1)
#         out = rearrange(out.transpose(-2, -1), "B C (H W) -> B C H W", H=H, W=W)

#         gate = self._edge_gate(x)
#         return x + self.out(out) * gate
class EdgeGateAdapter(nn.Module):
    def __init__(self, width: int, prior_in_channels: int = 256, ksize: int = 3):
        super().__init__()
        self.conv_in = nn.Conv2d(prior_in_channels, width, 1)

        self.save_attention = False
        self.attn_cache = {}

        self.sx = nn.Conv2d(1, 1, ksize, padding=ksize // 2, bias=False)
        self.sy = nn.Conv2d(1, 1, ksize, padding=ksize // 2, bias=False)
        with torch.no_grad():
            sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            if self.sx.weight.shape == sx.shape:
                self.sx.weight.copy_(sx)
                self.sy.weight.copy_(sy)

        self.get_kv = nn.Sequential(LayerNorm(width), nn.Conv2d(width, width * 2, 1))
        self.get_q = nn.Sequential(LayerNorm(width), nn.Conv2d(width, width, 1))
        self.out = ZeroConv2d(width, width)

    def _edge_gate(self, x: torch.Tensor):
        gimg = x.mean(1, keepdim=True)
        ex, ey = self.sx(gimg), self.sy(gimg)
        mag = torch.sqrt(ex**2 + ey**2)
        gate = torch.sigmoid(mag.mean(dim=[1, 2, 3], keepdim=True))
        return gate

    def forward(self, x: torch.Tensor, prior: torch.Tensor):
        prior = F.interpolate(self.conv_in(prior), size=x.shape[2:], mode="bilinear", align_corners=False)
        k, v = self.get_kv(prior).chunk(2, dim=1)
        q = self.get_q(x)

        B, C, H, W = x.shape
        q_ = rearrange(q, "B C H W -> B C (H W)")
        k_ = rearrange(k, "B C H W -> B C (H W)")
        v_ = rearrange(v, "B C H W -> B C (H W)")

        attn_logits = (q_.transpose(-2, -1) @ k_) / (C ** 0.5)
        attn = attn_logits.softmax(dim=-1)

        out = attn @ v_.transpose(-2, -1)
        out = rearrange(out.transpose(-2, -1), "B C (H W) -> B C H W", H=H, W=W)

        gate = self._edge_gate(x)
        residual = self.out(out) * gate
        final = x + residual

        if self.save_attention:
            self.attn_cache = {
                "x_before_q": x.detach(),
                "pre_kv_feature": prior.detach(),
                "q": q.detach(),
                "k": k.detach(),
                "v": v.detach(),
                "attn_logits": attn_logits.detach(),
                "attn_softmax": attn.detach(),
                "edge_gate": gate.detach(),
                "residual_branch": residual.detach(),
                "out_after_residual": final.detach(),
            }

        return final




class UNet2DConditionModelAdapter(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: int = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        attention_type: str = "default",
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        addition_embed_type_num_heads=64,

        # =========================
        # Ablation toggles (NEW)
        # =========================
        use_adapter: bool = True,          # 总开关
        use_down_adapter: bool = False,     # down stage adapter 开关
        use_mid_adapter: bool = True,      # mid stage adapter 开关
        use_up_adapter: bool = True,       # up stage adapter 开关
        down_adapter_mask: Optional[Tuple[bool, ...]] = None,  # 长度=down blocks数；None->全开
        up_adapter_mask: Optional[Tuple[bool, ...]] = None,    # 长度=up blocks数；None->全开
        

    ):
        super().__init__()

        self.sample_size = sample_size
        self.save_attention_maps = False  # New 

        # ===== runtime flags (can be changed after loading)
        self.use_adapter = bool(use_adapter)
        self.use_down_adapter = bool(use_down_adapter)
        self.use_mid_adapter = bool(use_mid_adapter)
        self.use_up_adapter = bool(use_up_adapter)

        if num_attention_heads is not None:
            raise ValueError(
                "At the moment it is not possible to define the number of attention heads via `num_attention_heads` "
                "because of a naming issue in diffusers. Passing `num_attention_heads` will only be supported in diffusers v0.19."
            )

        num_attention_heads = num_attention_heads or attention_head_dim

        if len(down_block_types) != len(up_block_types):
            raise ValueError(f"Must provide the same number of `down_block_types` as `up_block_types`.")
        if len(block_out_channels) != len(down_block_types):
            raise ValueError(f"Must provide the same number of `block_out_channels` as `down_block_types`.")
        if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
            raise ValueError(f"Must provide the same number of `only_cross_attention` as `down_block_types`.")
        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(f"Must provide the same number of `num_attention_heads` as `down_block_types`.")
        if not isinstance(attention_head_dim, int) and len(attention_head_dim) != len(down_block_types):
            raise ValueError(f"Must provide the same number of `attention_head_dim` as `down_block_types`.")
        if isinstance(cross_attention_dim, list) and len(cross_attention_dim) != len(down_block_types):
            raise ValueError(f"Must provide the same number of `cross_attention_dim` as `down_block_types`.")
        if not isinstance(layers_per_block, int) and len(layers_per_block) != len(down_block_types):
            raise ValueError(f"Must provide the same number of `layers_per_block` as `down_block_types`.")
        if isinstance(transformer_layers_per_block, list) and reverse_transformer_layers_per_block is None:
            for layer_number_per_block in transformer_layers_per_block:
                if isinstance(layer_number_per_block, list):
                    raise ValueError("Must provide 'reverse_transformer_layers_per_block` if using asymmetrical UNet.")

        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding)

        # time
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}.")
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 4
            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        else:
            raise ValueError(f"{time_embedding_type} does not exist. Use `fourier` or `positional`.")

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim, time_embed_dim, act_fn=act_fn, post_act_fn=timestep_post_act, cond_proj_dim=time_cond_proj_dim
        )

        if encoder_hid_dim_type is None and encoder_hid_dim is not None:
            encoder_hid_dim_type = "text_proj"
            self.register_to_config(encoder_hid_dim_type=encoder_hid_dim_type)
            logger.info("encoder_hid_dim_type defaults to 'text_proj' as `encoder_hid_dim` is defined.")

        if encoder_hid_dim is None and encoder_hid_dim_type is not None:
            raise ValueError(f"`encoder_hid_dim` has to be defined when `encoder_hid_dim_type` is set.")

        if encoder_hid_dim_type == "text_proj":
            self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim)
        elif encoder_hid_dim_type == "text_image_proj":
            self.encoder_hid_proj = TextImageProjection(
                text_embed_dim=encoder_hid_dim, image_embed_dim=cross_attention_dim, cross_attention_dim=cross_attention_dim
            )
        elif encoder_hid_dim_type == "image_proj":
            self.encoder_hid_proj = ImageProjection(image_embed_dim=encoder_hid_dim, cross_attention_dim=cross_attention_dim)
        elif encoder_hid_dim_type is not None:
            raise ValueError("encoder_hid_dim_type must be None, 'text_proj' or 'text_image_proj'.")
        else:
            self.encoder_hid_proj = None

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim, act_fn=act_fn)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        elif class_embed_type == "projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError("`class_embed_type='projection'` requires `projection_class_embeddings_input_dim`.")
            self.class_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        elif class_embed_type == "simple_projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError("`class_embed_type='simple_projection'` requires `projection_class_embeddings_input_dim`.")
            self.class_embedding = nn.Linear(projection_class_embeddings_input_dim, time_embed_dim)
        else:
            self.class_embedding = None

        if addition_embed_type == "text":
            text_time_embedding_from_dim = encoder_hid_dim if encoder_hid_dim is not None else cross_attention_dim
            self.add_embedding = TextTimeEmbedding(text_time_embedding_from_dim, time_embed_dim, num_heads=addition_embed_type_num_heads)
        elif addition_embed_type == "text_image":
            self.add_embedding = TextImageTimeEmbedding(
                text_embed_dim=cross_attention_dim, image_embed_dim=cross_attention_dim, time_embed_dim=time_embed_dim
            )
        elif addition_embed_type == "text_time":
            self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos, freq_shift)
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        elif addition_embed_type == "image":
            self.add_embedding = ImageTimeEmbedding(image_embed_dim=encoder_hid_dim, time_embed_dim=time_embed_dim)
        elif addition_embed_type == "image_hint":
            self.add_embedding = ImageHintTimeEmbedding(image_embed_dim=encoder_hid_dim, time_embed_dim=time_embed_dim)
        elif addition_embed_type is not None:
            raise ValueError("addition_embed_type must be None, 'text' or 'text_image'.")

        self.time_embed_act = None if time_embedding_act_fn is None else get_activation(time_embedding_act_fn)

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            if mid_block_only_cross_attention is None:
                mid_block_only_cross_attention = only_cross_attention
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if mid_block_only_cross_attention is None:
            mid_block_only_cross_attention = False

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)
        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)
        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)
        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        blocks_time_embed_dim = time_embed_dim * 2 if class_embeddings_concat else time_embed_dim

        # =========================
        # adapters
        # =========================
        self.down_adapter_blocks = nn.ModuleList([])
        self.up_adapter_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            self.down_adapter_blocks.append(FreqSplitAdapter(output_channel, prior_in_channels=256, stage="down"))

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=num_attention_heads[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_type=attention_type,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                cross_attention_norm=cross_attention_norm,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

        # mid
        if mid_block_type == "UNetMidBlock2DCrossAttn":
            self.mid_block = UNetMidBlock2DCrossAttn(
                transformer_layers_per_block=transformer_layers_per_block[-1],
                in_channels=block_out_channels[-1],
                temb_channels=blocks_time_embed_dim,
                dropout=dropout,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim[-1],
                num_attention_heads=num_attention_heads[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
                attention_type=attention_type,
            )
        elif mid_block_type == "UNetMidBlock2DSimpleCrossAttn":
            self.mid_block = UNetMidBlock2DSimpleCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=blocks_time_embed_dim,
                dropout=dropout,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                cross_attention_dim=cross_attention_dim[-1],
                attention_head_dim=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                resnet_time_scale_shift=resnet_time_scale_shift,
                skip_time_act=resnet_skip_time_act,
                only_cross_attention=mid_block_only_cross_attention,
                cross_attention_norm=cross_attention_norm,
            )
        elif mid_block_type == "UNetMidBlock2D":
            self.mid_block = UNetMidBlock2D(
                in_channels=block_out_channels[-1],
                temb_channels=blocks_time_embed_dim,
                dropout=dropout,
                num_layers=0,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_groups=norm_num_groups,
                resnet_time_scale_shift=resnet_time_scale_shift,
                add_attention=False,
            )
        elif mid_block_type is None:
            self.mid_block = None
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")

        self.mid_adapter_blocks = PyramidAdapter(block_out_channels[-1], prior_in_channels=256, stage="mid")

        # count upsamplers
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = (
            list(reversed(transformer_layers_per_block))
            if reverse_transformer_layers_per_block is None
            else reverse_transformer_layers_per_block
        )
        only_cross_attention = list(reversed(only_cross_attention))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            self.up_adapter_blocks.append(EdgeGateAdapter(output_channel, prior_in_channels=256))

            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=blocks_time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resolution_idx=i,
                resnet_groups=norm_num_groups,
                cross_attention_dim=reversed_cross_attention_dim[i],
                num_attention_heads=reversed_num_attention_heads[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_type=attention_type,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                cross_attention_norm=cross_attention_norm,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                dropout=dropout,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
            self.conv_act = get_activation(act_fn)
        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding)

        if attention_type in ["gated", "gated-text-image"]:
            positive_len = 768
            if isinstance(cross_attention_dim, int):
                positive_len = cross_attention_dim
            elif isinstance(cross_attention_dim, (tuple, list)):
                positive_len = cross_attention_dim[0]

            feature_type = "text-only" if attention_type == "gated" else "text-image"
            self.position_net = PositionNet(positive_len=positive_len, out_dim=cross_attention_dim, feature_type=feature_type)

        # ===== init runtime masks (NOTE: stored separately from config tuple)
        self.down_adapter_mask = self._normalize_mask(down_adapter_mask, len(self.down_adapter_blocks), "down_adapter_mask")
        self.up_adapter_mask = self._normalize_mask(up_adapter_mask, len(self.up_adapter_blocks), "up_adapter_mask")


    def enable_adapter_attention_save(self, enabled: bool = True, stage: str = "mid"):
        stage = str(stage).lower()
        assert stage in ["mid", "down", "up", "all"]

        self.save_attention_maps = enabled

        for m in self.down_adapter_blocks:
            m.save_attention = bool(enabled and stage in ["down", "all"])

        self.mid_adapter_blocks.save_attention = bool(enabled and stage in ["mid", "all"])

        for m in self.up_adapter_blocks:
            m.save_attention = bool(enabled and stage in ["up", "all"])


    def collect_adapter_attention(self):
        data = {
            "down": [],
            "mid": None,
            "up": [],
        }
        for m in self.down_adapter_blocks:
            data["down"].append(getattr(m, "attn_cache", {}))
        data["mid"] = getattr(self.mid_adapter_blocks, "attn_cache", {})
        for m in self.up_adapter_blocks:
            data["up"].append(getattr(m, "attn_cache", {}))
        return data


    def clear_adapter_attention(self):
        for m in self.down_adapter_blocks:
            m.attn_cache = {}
        self.mid_adapter_blocks.attn_cache = {}
        for m in self.up_adapter_blocks:
            m.attn_cache = {}



    # =========================
    # Ablation helpers (NEW)
    # =========================
    @staticmethod
    def _normalize_mask(mask: Optional[Union[List[bool], Tuple[bool, ...]]], n: int, name: str) -> List[bool]:
        if mask is None:
            return [True] * n
        m = list(mask)
        if len(m) != n:
            raise ValueError(f"{name} length mismatch: got {len(m)} but expected {n}")
        return [bool(x) for x in m]

    def set_adapter_config(
        self,
        *,
        use_adapter: Optional[bool] = None,
        use_down_adapter: Optional[bool] = None,
        use_mid_adapter: Optional[bool] = None,
        use_up_adapter: Optional[bool] = None,
        down_adapter_mask: Optional[Union[List[bool], Tuple[bool, ...]]] = None,
        up_adapter_mask: Optional[Union[List[bool], Tuple[bool, ...]]] = None,
    ):
        """运行时修改 ablation 开关（不需要重建模型）"""
        if use_adapter is not None:
            self.use_adapter = bool(use_adapter)
        if use_down_adapter is not None:
            self.use_down_adapter = bool(use_down_adapter)
        if use_mid_adapter is not None:
            self.use_mid_adapter = bool(use_mid_adapter)
        if use_up_adapter is not None:
            self.use_up_adapter = bool(use_up_adapter)
        if down_adapter_mask is not None:
            self.down_adapter_mask = self._normalize_mask(down_adapter_mask, len(self.down_adapter_blocks), "down_adapter_mask")
        if up_adapter_mask is not None:
            self.up_adapter_mask = self._normalize_mask(up_adapter_mask, len(self.up_adapter_blocks), "up_adapter_mask")
        return self

    def set_adapter_mask(self, stage: str, mask: Union[List[bool], Tuple[bool, ...]]):
        """只改某个 stage 的逐-block mask"""
        stage = stage.lower().strip()
        if stage == "down":
            self.down_adapter_mask = self._normalize_mask(mask, len(self.down_adapter_blocks), "down_adapter_mask")
        elif stage == "up":
            self.up_adapter_mask = self._normalize_mask(mask, len(self.up_adapter_blocks), "up_adapter_mask")
        else:
            raise ValueError("stage must be 'down' or 'up'")
        return self

    def freeze_adapters(self, freeze: bool = True):
        """可选：冻结/解冻 adapter 参数（做公平对比时常用）"""
        modules = [self.down_adapter_blocks, self.mid_adapter_blocks, self.up_adapter_blocks]
        for m in modules:
            for p in m.parameters():
                p.requires_grad = (not freeze)
        return self

    # =========================
    # diffusers default APIs
    # =========================
    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)
            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)
        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]], _remove_lora=False):
        count = len(self.attn_processors.keys())
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match "
                f"the number of attention layers: {count}."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor, _remove_lora=_remove_lora)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"), _remove_lora=_remove_lora)
            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError("Cannot call `set_default_attn_processor` with mixed attention processor types.")
        self.set_attn_processor(processor, _remove_lora=True)

    def set_attention_slice(self, slice_size):
        sliceable_head_dims = []

        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)
            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            slice_size = num_sliceable_layers * [1]

        slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(f"Provided {len(slice_size)} slices, but model has {len(sliceable_head_dims)} layers.")

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())
            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def enable_freeu(self, s1, s2, b1, b2):
        for upsample_block in self.up_blocks:
            setattr(upsample_block, "s1", s1)
            setattr(upsample_block, "s2", s2)
            setattr(upsample_block, "b1", b1)
            setattr(upsample_block, "b2", b2)

    def disable_freeu(self):
        freeu_keys = {"s1", "s2", "b1", "b2"}
        for upsample_block in self.up_blocks:
            for k in freeu_keys:
                if hasattr(upsample_block, k) or getattr(upsample_block, k, None) is not None:
                    setattr(upsample_block, k, None)

    def fuse_qkv_projections(self):
        self.original_attn_processors = None
        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` not supported for models having added KV projections.")
        self.original_attn_processors = self.attn_processors
        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

    def unfuse_qkv_projections(self):
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def forward(
        self,
        sample: torch.FloatTensor,
        prior: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        default_overall_up_factor = 2 ** self.num_upsamplers

        forward_upsample_size = False
        upsample_size = None
        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                forward_upsample_size = True
                break

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
        elif self.config.addition_embed_type == "text_image":
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError("addition_embed_type='text_image' requires `image_embeds` in added_cond_kwargs")
            image_embs = added_cond_kwargs.get("image_embeds")
            text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
            aug_emb = self.add_embedding(text_embs, image_embs)
        elif self.config.addition_embed_type == "text_time":
            if "text_embeds" not in added_cond_kwargs or "time_ids" not in added_cond_kwargs:
                raise ValueError("addition_embed_type='text_time' requires `text_embeds` and `time_ids`")
            text_embeds = added_cond_kwargs.get("text_embeds")
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1).to(emb.dtype)
            aug_emb = self.add_embedding(add_embeds)
        elif self.config.addition_embed_type == "image":
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError("addition_embed_type='image' requires `image_embeds`")
            image_embs = added_cond_kwargs.get("image_embeds")
            aug_emb = self.add_embedding(image_embs)
        elif self.config.addition_embed_type == "image_hint":
            if "image_embeds" not in added_cond_kwargs or "hint" not in added_cond_kwargs:
                raise ValueError("addition_embed_type='image_hint' requires `image_embeds` and `hint`")
            image_embs = added_cond_kwargs.get("image_embeds")
            hint = added_cond_kwargs.get("hint")
            aug_emb, hint = self.add_embedding(image_embs, hint)
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError("encoder_hid_dim_type='text_image_proj' requires `image_embeds`")
            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "image_proj":
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError("encoder_hid_dim_type='image_proj' requires `image_embeds`")
            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(image_embeds)

        # 2. pre-process
        sample = self.conv_in(sample)

        # GLIGEN
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        # 3. down
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0
        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        is_adapter = down_intrablock_additional_residuals is not None

        if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
            deprecate(
                "T2I should not use down_block_additional_residuals",
                "1.3.0",
                "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated "
                "and will be removed in diffusers 1.3.0. Use `down_intrablock_additional_residuals` instead.",
                standard_warn=False,
            )
            down_intrablock_additional_residuals = down_block_additional_residuals
            is_adapter = True

        down_block_res_samples = (sample,)

        for idx, (downsample_block, down_adapter_block) in enumerate(zip(self.down_blocks, self.down_adapter_blocks)):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                additional_residuals = {}
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, scale=lora_scale)
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    sample += down_intrablock_additional_residuals.pop(0)

            # ===== apply DOWN adapter (ablation-aware)
            if (
                self.use_adapter
                and self.use_down_adapter
                and self.down_adapter_mask[idx]
            ):
                sample = down_adapter_block(sample, prior)

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()
            for down_block_res_sample, down_block_additional_residual in zip(down_block_res_samples, down_block_additional_residuals):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)
            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.mid_block(sample, emb)

            if (
                is_adapter
                and len(down_intrablock_additional_residuals) > 0
                and sample.shape == down_intrablock_additional_residuals[0].shape
            ):
                sample += down_intrablock_additional_residuals.pop(0)

            # ===== apply MID adapter (ablation-aware)
            if self.use_adapter and self.use_mid_adapter:
                sample = self.mid_adapter_blocks(sample, prior)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, (upsample_block, up_adapter_block) in enumerate(zip(self.up_blocks, self.up_adapter_blocks)):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    scale=lora_scale,
                )

            # ===== apply UP adapter (ablation-aware)
            if (
                self.use_adapter
                and self.use_up_adapter
                and self.up_adapter_mask[i]
            ):
                sample = up_adapter_block(sample, prior)

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)

        sample = self.conv_out(sample)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)
