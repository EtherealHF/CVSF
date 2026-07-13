# ============================================
# File: src/cvsf.py
# Paper-aligned version:
# current guidance -> c
# c -> LoRA modulation
# c -> spatial prior for adapter
#
# Supports:
#   - c_guidance_type: image / text / depth / edge / seg
#   - shuffle_lora_in_batch
#   - shuffle_adapter_in_batch
#   - force_zero_c
#   - force_noise_c
#   - force_text_c
#   - force_zero_prior
# ============================================

import os
import re
import sys
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL
from peft import LoraConfig

from src.my_utils.stable_unet_adapter import UNet2DConditionModelAdapter

p = "src/"
sys.path.append(p)

from model import make_1step_sched, my_lora_fwd
from basicsr.archs.arch_util import default_init_weights
from net_utils_ablation import WaveletPriorExtraction


def get_layer_number(module_name):
    base_layers = {
        "down_blocks": 0,
        "mid_block": 4,
        "up_blocks": 5,
    }

    if module_name == "conv_out":
        return 9

    base_layer = None
    for key in base_layers:
        if key in module_name:
            base_layer = base_layers[key]
            break

    if base_layer is None:
        return None

    additional_layers = int(re.findall(r"\.(\d+)", module_name)[0])
    final_layer = base_layer + additional_layers
    return final_layer


class CVSF(torch.nn.Module):
    def __init__(
        self,
        sd_path=None,
        pretrained_path=None,
        lora_rank_unet=32,
        lora_rank_vae=32,
        block_embedding_dim=64,
        ablation_mode: str = "full",
        lora_zero_parts: bool = False,

        # ========= Stage1 image encoder =========
        stage1_img_enc: Optional[nn.Module] = None,
        freeze_stage1_img_enc: bool = True,
        aligned_dim: int = 1024,

        # ========= prior map =========
        prior_map_size: int = 8,
        prior_channels: int = 256,

        # ========= seg input channels =========
        seg_in_channels: int = 1,
    ):
        super().__init__()

        assert ablation_mode in [
            "full",
            "no_eeg_all",
            "no_eeg_lora",
            "no_eeg_adapter",
            "no_lora",
            "vanilla_lora",
        ]
        self.ablation_mode = ablation_mode

        self.enable_lora = (self.ablation_mode != "no_lora")
        self.enable_dynamic_lora = self.ablation_mode not in ["no_lora", "vanilla_lora"]

        self.lora_rank_unet = lora_rank_unet
        self.lora_rank_vae = lora_rank_vae
        self.lora_zero_parts = bool(lora_zero_parts)

        self.stage1_img_enc = stage1_img_enc
        self.aligned_dim = aligned_dim
        self.prior_channels = prior_channels
        self.prior_map_size = prior_map_size
        self.seg_in_channels = seg_in_channels

        if self.stage1_img_enc is not None and freeze_stage1_img_enc:
            self.stage1_img_enc.eval()
            for p in self.stage1_img_enc.parameters():
                p.requires_grad = False

        # ------------------------------
        # 1. text / SD backbone
        # ------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").cuda()
        self.text_encoder.requires_grad_(False)

        self.sched = make_1step_sched(sd_path)

        vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
        unet = UNet2DConditionModelAdapter.from_pretrained(
            sd_path,
            subfolder="unet",
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
        )

        if self.ablation_mode in ["no_eeg_all", "no_eeg_adapter", "vanilla_lora"]:
            unet.use_adapter = False
        else:
            unet.use_adapter = True

        self.target_modules_vae = r"^encoder\..*(conv1|conv2|conv_in|conv_shortcut|conv|conv_out|to_k|to_q|to_v|to_out\.0)$"
        self.target_modules_unet = [
            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
            "conv_shortcut", "conv_out", "proj_in", "proj_out",
            "ff.net.2", "ff.net.0.proj"
        ]

        # ------------------------------
        # 2. condition blocks
        # ------------------------------
        self.wpenet = WaveletPriorExtraction()

        num_embeddings = 64
        self.W = nn.Parameter(torch.randn(num_embeddings), requires_grad=False)

        # c -> 256, for LoRA cognition condition
        self.eeg_mlp = nn.Sequential(
            nn.Linear(self.aligned_dim, 256),
            nn.ReLU(True),
        )

        # c -> [B, C, H, W], for adapter prior
        self.prior_mapper = nn.Sequential(
            nn.Linear(self.aligned_dim, self.prior_channels * self.prior_map_size * self.prior_map_size),
            nn.ReLU(True),
        )

        # text pooled feature -> aligned_dim
        text_hidden_dim = self.text_encoder.config.hidden_size
        if text_hidden_dim == self.aligned_dim:
            self.text_c_proj = nn.Identity()
        else:
            self.text_c_proj = nn.Sequential(
                nn.Linear(text_hidden_dim, self.aligned_dim),
                nn.ReLU(True),
            )

        # depth -> c
        self.depth_c_net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, self.aligned_dim),
        )

        # edge -> c
        self.edge_c_net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, self.aligned_dim),
        )

        # seg -> c
        self.seg_c_net = nn.Sequential(
            nn.Conv2d(self.seg_in_channels, 32, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, self.aligned_dim),
        )

        self.vae_de_mlp_1 = nn.Sequential(
            nn.Linear(num_embeddings * 2, 256),
            nn.ReLU(True),
        )
        self.vae_de_mlp_2 = nn.Sequential(
            nn.Linear(num_embeddings * 2, 256),
            nn.ReLU(True),
        )

        self.vae_block_mlp = nn.Sequential(
            nn.Linear(block_embedding_dim, 64),
            nn.ReLU(True),
        )

        self.unet_processor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(start_dim=1),
            nn.Linear(16 * 8 * 8, 16 * (lora_rank_vae ** 2)),
        )

        self.vae_fuse_mlp_0 = nn.Linear(lora_rank_vae ** 2 + 64 + 256, lora_rank_vae ** 2)
        self.vae_fuse_mlp_1 = nn.Linear(256 + 64 + 256, lora_rank_vae ** 2)
        self.vae_fuse_mlp_2 = nn.Linear(256 + 64 + 256, lora_rank_vae ** 2)

        init_modules = [
            self.eeg_mlp,
            self.prior_mapper,
            self.depth_c_net,
            self.edge_c_net,
            self.seg_c_net,
            self.vae_de_mlp_1,
            self.vae_de_mlp_2,
            self.vae_block_mlp,
            self.vae_fuse_mlp_0,
            self.vae_fuse_mlp_1,
            self.vae_fuse_mlp_2,
            self.unet_processor,
            self.wpenet,
        ]

        if not isinstance(self.text_c_proj, nn.Identity):
            init_modules.append(self.text_c_proj)

        default_init_weights(init_modules, 1e-5)

        self.vae_block_embeddings = nn.Embedding(16, block_embedding_dim)

        # ------------------------------
        # 3. load pretrained / add lora
        # ------------------------------
        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")

            if self.enable_lora:
                vae_lora_config = LoraConfig(
                    r=sd["rank_vae"],
                    init_lora_weights="gaussian",
                    target_modules=sd["vae_lora_target_modules"],
                )
                vae.add_adapter(vae_lora_config, adapter_name="vae_skip_0")
                vae.add_adapter(vae_lora_config, adapter_name="vae_skip_1")
                vae.add_adapter(vae_lora_config, adapter_name="vae_skip_2")
                vae.set_adapter(["vae_skip_0", "vae_skip_1", "vae_skip_2"])

                unet_lora_config = LoraConfig(
                    r=sd["rank_unet"],
                    init_lora_weights="gaussian",
                    target_modules=sd["unet_lora_target_modules"],
                )
                unet.add_adapter(unet_lora_config, adapter_name="unet_skip_0")
                unet.add_adapter(unet_lora_config, adapter_name="unet_skip_1")
                unet.add_adapter(unet_lora_config, adapter_name="unet_skip_2")
                unet.set_adapter(["unet_skip_0", "unet_skip_1", "unet_skip_2"])

            _sd_vae = vae.state_dict()
            for k, v in sd["state_dict_vae"].items():
                if (not self.enable_lora) and ("lora" in k):
                    continue
                _sd_vae[k] = v
            vae.load_state_dict(_sd_vae, strict=False)

            _sd_unet = unet.state_dict()
            for k, v in sd["state_dict_unet"].items():
                if (not self.enable_lora) and ("lora" in k):
                    continue
                _sd_unet[k] = v
            unet.load_state_dict(_sd_unet, strict=False)

            def safe_load(module, key):
                if key not in sd:
                    return
                cur = module.state_dict()
                for k, v in sd[key].items():
                    if k in cur and cur[k].shape == v.shape:
                        cur[k] = v
                module.load_state_dict(cur, strict=False)

            safe_load(self.vae_de_mlp_1, "state_dict_vae_de_mlp_1")
            safe_load(self.vae_de_mlp_2, "state_dict_vae_de_mlp_2")
            safe_load(self.vae_block_mlp, "state_dict_vae_block_mlp")
            safe_load(self.vae_fuse_mlp_0, "state_dict_vae_fuse_mlp_0")
            safe_load(self.vae_fuse_mlp_1, "state_dict_vae_fuse_mlp_1")
            safe_load(self.vae_fuse_mlp_2, "state_dict_vae_fuse_mlp_2")
            safe_load(self.unet_processor, "state_dict_unet_processor")
            safe_load(self.wpenet, "state_dict_wpenet")
            safe_load(self.eeg_mlp, "state_dict_eeg_mlp")
            safe_load(self.prior_mapper, "state_dict_prior_mapper")

            if not isinstance(self.text_c_proj, nn.Identity):
                safe_load(self.text_c_proj, "state_dict_text_c_proj")

            safe_load(self.depth_c_net, "state_dict_depth_c_net")
            safe_load(self.edge_c_net, "state_dict_edge_c_net")
            safe_load(self.seg_c_net, "state_dict_seg_c_net")

            if "w" in sd:
                self.W = nn.Parameter(sd["w"], requires_grad=False)

            if "state_embeddings" in sd and "state_dict_vae_block" in sd["state_embeddings"]:
                self.vae_block_embeddings.load_state_dict(sd["state_embeddings"]["state_dict_vae_block"])

        else:
            if self.enable_lora:
                vae_lora_config = LoraConfig(
                    r=lora_rank_vae,
                    init_lora_weights="gaussian",
                    target_modules=self.target_modules_vae,
                )
                vae.add_adapter(vae_lora_config, adapter_name="vae_skip_0")
                vae.add_adapter(vae_lora_config, adapter_name="vae_skip_1")
                vae.add_adapter(vae_lora_config, adapter_name="vae_skip_2")
                vae.set_adapter(["vae_skip_0", "vae_skip_1", "vae_skip_2"])

                unet_lora_config = LoraConfig(
                    r=lora_rank_unet,
                    init_lora_weights="gaussian",
                    target_modules=self.target_modules_unet,
                )
                unet.add_adapter(unet_lora_config, adapter_name="unet_skip_0")
                unet.add_adapter(unet_lora_config, adapter_name="unet_skip_1")
                unet.add_adapter(unet_lora_config, adapter_name="unet_skip_2")
                unet.set_adapter(["unet_skip_0", "unet_skip_1", "unet_skip_2"])

        # ------------------------------
        # 4. dynamic LoRA
        # ------------------------------
        self.vae_lora_layers = []
        self.unet_lora_layers = []

        if self.enable_dynamic_lora:
            for name, module in vae.named_modules():
                if "base_layer" in name:
                    self.vae_lora_layers.append(name[:-len(".base_layer")])
            for name, module in vae.named_modules():
                if name in self.vae_lora_layers:
                    module.forward = my_lora_fwd.__get__(module, module.__class__)

            for name, module in unet.named_modules():
                if "base_layer" in name:
                    self.unet_lora_layers.append(name[:-len(".base_layer")])
            for name, module in unet.named_modules():
                if name in self.unet_lora_layers:
                    module.forward = my_lora_fwd.__get__(module, module.__class__)

        unet.to("cuda")
        vae.to("cuda")
        self.unet, self.vae = unet, vae
        self.timesteps = torch.tensor([49], device="cuda").long()

    # ----------------------------------------
    # mode
    # ----------------------------------------
    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.vae_de_mlp_1.eval()
        self.vae_de_mlp_2.eval()
        self.vae_block_mlp.eval()
        self.vae_fuse_mlp_0.eval()
        self.vae_fuse_mlp_1.eval()
        self.vae_fuse_mlp_2.eval()
        self.unet_processor.eval()
        self.eeg_mlp.eval()
        self.prior_mapper.eval()
        self.text_c_proj.eval()
        self.depth_c_net.eval()
        self.edge_c_net.eval()
        self.seg_c_net.eval()
        self.wpenet.eval()

        if self.stage1_img_enc is not None:
            self.stage1_img_enc.eval()
            for p in self.stage1_img_enc.parameters():
                p.requires_grad = False

        self.vae_block_embeddings.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet_processor.requires_grad_(False)
        self.wpenet.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.vae_de_mlp_1.train()
        self.vae_de_mlp_2.train()
        self.vae_block_mlp.train()
        self.vae_fuse_mlp_0.train()
        self.vae_fuse_mlp_1.train()
        self.vae_fuse_mlp_2.train()
        self.unet_processor.train()
        self.eeg_mlp.train()
        self.prior_mapper.train()
        self.text_c_proj.train()
        self.depth_c_net.train()
        self.edge_c_net.train()
        self.seg_c_net.train()
        self.wpenet.train()

        if self.stage1_img_enc is not None:
            self.stage1_img_enc.eval()
            for p in self.stage1_img_enc.parameters():
                p.requires_grad = False

        self.vae_block_embeddings.requires_grad_(True)

        for n, _p in self.unet.named_parameters():
            if "adapter_blocks" in n or "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)

        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

    # ----------------------------------------
    # helper
    # ----------------------------------------
    def _get_aligned_embedding(self, c_t):
        if self.stage1_img_enc is None:
            raise RuntimeError("stage1_img_enc is required for image-guidance c.")
        with torch.no_grad():
            c = self.stage1_img_enc(c_t)
        if c.dim() != 2:
            raise ValueError(f"stage1_img_enc output must be [B, D], got {tuple(c.shape)}")
        return c.to(device=c_t.device, dtype=c_t.dtype)

    def _make_prior_from_c(self, c, B, device, dtype):
        prior = self.prior_mapper(c)
        prior = prior.view(B, self.prior_channels, self.prior_map_size, self.prior_map_size)
        return prior.to(device=device, dtype=dtype)

    def _make_perm(self, B, device, seed=None):
        if seed is None:
            return torch.randperm(B, device=device)
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
        return torch.randperm(B, generator=g, device=device)

    def _make_noise_c(self, ref_c, seed=None, scale=1.0, match_stats=False):
        device = ref_c.device
        dtype = ref_c.dtype

        if seed is None:
            noise = torch.randn_like(ref_c) * scale
        else:
            g = torch.Generator(device=device)
            g.manual_seed(int(seed))
            noise = torch.randn(ref_c.shape, generator=g, device=device, dtype=dtype) * scale

        if match_stats:
            mean = ref_c.mean(dim=1, keepdim=True)
            std = ref_c.std(dim=1, keepdim=True).clamp_min(1e-6)
            noise_mean = noise.mean(dim=1, keepdim=True)
            noise_std = noise.std(dim=1, keepdim=True).clamp_min(1e-6)
            noise = (noise - noise_mean) / noise_std
            noise = noise * std + mean

        return noise

    def _make_band_noise_c(
        self,
        ref_c,
        seed=None,
        scale=1.0,
        band_start_ratio=0.2,
        band_end_ratio=0.5,
        match_stats=False,
    ):
        """
        ref_c: [B, D]
        在频域的指定频段加入噪声，再 inverse FFT 回到时域/特征域
        """
        if ref_c.dim() != 2:
            raise ValueError(f"ref_c must be [B, D], got shape={ref_c.shape}")

        device = ref_c.device
        dtype = ref_c.dtype
        B, D = ref_c.shape

        if not (0.0 <= band_start_ratio < band_end_ratio <= 1.0):
            raise ValueError("band ratios must satisfy 0 <= start < end <= 1")

        # 生成基础噪声
        if seed is None:
            noise = torch.randn(B, D, device=device, dtype=torch.float32)
        else:
            g = torch.Generator(device=device)
            g.manual_seed(int(seed))
            noise = torch.randn(B, D, generator=g, device=device, dtype=torch.float32)

        # FFT 到频域
        noise_fft = torch.fft.rfft(noise, dim=1)   # [B, D_rfft]
        Fdim = noise_fft.shape[1]

        start_idx = max(0, min(Fdim - 1, int(Fdim * band_start_ratio)))
        end_idx = max(start_idx + 1, min(Fdim, int(Fdim * band_end_ratio)))

        # 构造频段 mask，只保留指定频段
        mask = torch.zeros(Fdim, device=device, dtype=torch.float32)
        mask[start_idx:end_idx] = 1.0
        mask = mask.view(1, Fdim)

        band_fft = noise_fft * mask

        # 回到特征域
        band_noise = torch.fft.irfft(band_fft, n=D, dim=1)

        # 控制强度
        band_noise = band_noise * float(scale)

        # 可选：匹配原 c 的统计量
        if match_stats:
            ref_mean = ref_c.float().mean(dim=1, keepdim=True)
            ref_std = ref_c.float().std(dim=1, keepdim=True).clamp_min(1e-6)

            noise_mean = band_noise.mean(dim=1, keepdim=True)
            noise_std = band_noise.std(dim=1, keepdim=True).clamp_min(1e-6)

            band_noise = (band_noise - noise_mean) / noise_std
            band_noise = band_noise * ref_std + ref_mean

        return band_noise.to(dtype=dtype)

    def _make_text_c(self, caption_enc, pool="mean", detach=False):
        if pool == "cls":
            text_feat = caption_enc[:, 0, :]
        else:
            text_feat = caption_enc.mean(dim=1)

        c_text = self.text_c_proj(text_feat)
        if detach:
            c_text = c_text.detach()
        return c_text

    def _make_depth_c(self, depth_map):
        if depth_map is None:
            raise ValueError("depth_map is required when c_guidance_type='depth'")
        if depth_map.shape[1] != 1:
            if depth_map.shape[1] > 1:
                depth_map = depth_map[:, :1, :, :]
            else:
                raise ValueError(f"depth_map should be [B,1,H,W], got {tuple(depth_map.shape)}")
        return self.depth_c_net(depth_map)

    def _make_edge_c(self, edge_map):
        if edge_map is None:
            raise ValueError("edge_map is required when c_guidance_type='edge'")
        if edge_map.shape[1] != 1:
            if edge_map.shape[1] > 1:
                edge_map = edge_map[:, :1, :, :]
            else:
                raise ValueError(f"edge_map should be [B,1,H,W], got {tuple(edge_map.shape)}")
        return self.edge_c_net(edge_map)

    def _make_seg_c(self, seg_map):
        if seg_map is None:
            raise ValueError("seg_map is required when c_guidance_type='seg'")
        if seg_map.shape[1] != self.seg_in_channels:
            raise ValueError(
                f"seg_map channels mismatch: expect {self.seg_in_channels}, got {seg_map.shape[1]}"
            )
        return self.seg_c_net(seg_map)

    def _get_guidance_c(
        self,
        c_guidance_type,
        c_t,
        caption_enc,
        depth_map=None,
        edge_map=None,
        seg_map=None,
        text_c_pool="mean",
        text_c_detach=False,
    ):
        if c_guidance_type == "image":
            c = self._get_aligned_embedding(c_t)
        elif c_guidance_type == "text":
            c = self._make_text_c(caption_enc, pool=text_c_pool, detach=text_c_detach)
        elif c_guidance_type == "depth":
            c = self._make_depth_c(depth_map)
        elif c_guidance_type == "edge":
            c = self._make_edge_c(edge_map)
        elif c_guidance_type == "seg":
            c = self._make_seg_c(seg_map)
        else:
            raise ValueError(f"Unknown c_guidance_type: {c_guidance_type}")

        return c.to(device=c_t.device, dtype=c_t.dtype)

    # ----------------------------------------
    # forward
    # ----------------------------------------
    def forward(
        self,
        c_t,
        deg_score,
        prompt,
        eeg_override_mode=None,
        eeg_override_target=None,
        shuffle_lora_in_batch=False,
        shuffle_adapter_in_batch=False,
        shuffle_seed=None,
        force_zero_prior=False,
        force_zero_c=False,
        force_noise_c=False,
        noise_c_scale=1.0,
        noise_c_seed=None,
        noise_c_match_stats=False,
        force_text_c=False,
        text_c_pool="mean",
        text_c_detach=False,

        # new
        c_guidance_type="image",
        depth_map=None,
        edge_map=None,
        seg_map=None,

        force_band_noise_c=False,
        band_noise_c_scale=1.0,
        band_noise_c_seed=None,
        band_start_ratio=0.2,
        band_end_ratio=0.5,
        band_noise_match_stats=False,

        return_attn_vis=False


    ):
        if prompt is None:
            raise ValueError("prompt must not be None")

        caption_tokens = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(c_t.device)
        caption_enc = self.text_encoder(caption_tokens)[0]

        B = c_t.shape[0]

        # ------------------------------------------
        # 1) guidance -> c
        # ------------------------------------------
        if self.ablation_mode in ["no_eeg_all", "no_eeg_lora"]:
            c = None
            c_lora = None
            c_adapter = None
        else:
            # 兼容旧逻辑：force_text_c 优先
            if force_text_c:
                c = self._make_text_c(
                    caption_enc,
                    pool=text_c_pool,
                    detach=text_c_detach,
                ).to(device=c_t.device, dtype=c_t.dtype)

            else:
                c = self._get_guidance_c(
                    c_guidance_type=c_guidance_type,
                    c_t=c_t,
                    caption_enc=caption_enc,
                    depth_map=depth_map,
                    edge_map=edge_map,
                    seg_map=seg_map,
                    text_c_pool=text_c_pool,
                    text_c_detach=text_c_detach,
                )

            if force_zero_c:
                c = torch.zeros_like(c)

            elif force_noise_c:
                c = self._make_noise_c(
                    c,
                    seed=noise_c_seed,
                    scale=noise_c_scale,
                    match_stats=noise_c_match_stats,
                )

            elif force_band_noise_c:
                c = self._make_band_noise_c(
                    c,
                    seed=band_noise_c_seed,
                    scale=band_noise_c_scale,
                    band_start_ratio=band_start_ratio,
                    band_end_ratio=band_end_ratio,
                    match_stats=band_noise_match_stats,
                )

            c_lora = c
            c_adapter = c

            if B > 1 and shuffle_lora_in_batch:
                perm_lora = self._make_perm(B, c.device, shuffle_seed)
                c_lora = c[perm_lora]

            if B > 1 and shuffle_adapter_in_batch:
                adapter_seed = None if shuffle_seed is None else int(shuffle_seed) + 1
                perm_adapter = self._make_perm(B, c.device, adapter_seed)
                c_adapter = c[perm_adapter]

        # ------------------------------------------
        # 2) dynamic LoRA modulation
        # ------------------------------------------
        if self.enable_lora and self.enable_dynamic_lora:
            init_illu = torch.max(c_t, dim=1, keepdim=True)[0]
            vae_de_c_embed_0 = self.unet_processor(init_illu)
            vae_de_c_embed_0 = vae_de_c_embed_0.view(B, 16, self.lora_rank_vae ** 2)

            deg_score1, deg_score2 = torch.split(deg_score, 1, dim=1)

            deg_proj1 = deg_score1[..., None] * self.W[None, None, :] * 2 * np.pi
            deg_proj1 = torch.cat([torch.sin(deg_proj1), torch.cos(deg_proj1)], dim=-1)
            deg_proj1 = deg_proj1.squeeze(1)
            vae_de_c_embed_1 = self.vae_de_mlp_1(deg_proj1)

            deg_proj2 = deg_score2[..., None] * self.W[None, None, :] * 2 * np.pi
            deg_proj2 = torch.cat([torch.sin(deg_proj2), torch.cos(deg_proj2)], dim=-1)
            deg_proj2 = deg_proj2.squeeze(1)
            vae_de_c_embed_2 = self.vae_de_mlp_2(deg_proj2)

            vae_block_c_embeds = self.vae_block_mlp(self.vae_block_embeddings.weight)
            num_blocks = vae_block_c_embeds.shape[0]
            block_emb_exp = vae_block_c_embeds.unsqueeze(0).repeat(B, 1, 1)

            if self.ablation_mode in ["no_eeg_all", "no_eeg_lora"]:
                eeg_cond_exp = torch.zeros((B, num_blocks, 256), device=c_t.device, dtype=c_t.dtype)
            else:
                eeg_cond = self.eeg_mlp(c_lora)
                eeg_cond_exp = eeg_cond.unsqueeze(1).repeat(1, num_blocks, 1)

            fused_0 = torch.cat([vae_de_c_embed_0, block_emb_exp, eeg_cond_exp], dim=-1)
            lora_embeds_0 = self.vae_fuse_mlp_0(fused_0)

            deg1_rep = vae_de_c_embed_1.unsqueeze(1).repeat(1, num_blocks, 1)
            fused_1 = torch.cat([deg1_rep, block_emb_exp, eeg_cond_exp], dim=-1)
            lora_embeds_1 = self.vae_fuse_mlp_1(fused_1)

            deg2_rep = vae_de_c_embed_2.unsqueeze(1).repeat(1, num_blocks, 1)
            fused_2 = torch.cat([deg2_rep, block_emb_exp, eeg_cond_exp], dim=-1)
            lora_embeds_2 = self.vae_fuse_mlp_2(fused_2)

            vae_embeds_0 = lora_embeds_0[:, :6]
            vae_embeds_1 = lora_embeds_1[:, :6]
            vae_embeds_2 = lora_embeds_2[:, :6]

            unet_embeds_0 = lora_embeds_0[:, 7:]
            unet_embeds_1 = lora_embeds_1[:, 7:]
            unet_embeds_2 = lora_embeds_2[:, 7:]

            if self.lora_zero_parts:
                vae_embeds_0 = torch.zeros_like(vae_embeds_0)
                vae_embeds_1 = torch.zeros_like(vae_embeds_1)
                vae_embeds_2 = torch.zeros_like(vae_embeds_2)
                unet_embeds_0 = torch.zeros_like(unet_embeds_0)
                unet_embeds_1 = torch.zeros_like(unet_embeds_1)
                unet_embeds_2 = torch.zeros_like(unet_embeds_2)

            for layer_name, module in self.vae.named_modules():
                if layer_name in self.vae_lora_layers:
                    split_name = layer_name.split(".")
                    if split_name[1] == "down_blocks":
                        block_id = int(split_name[2])
                        vae_embed_0 = vae_embeds_0[:, block_id]
                        vae_embed_1 = vae_embeds_1[:, block_id]
                        vae_embed_2 = vae_embeds_2[:, block_id]
                    elif split_name[1] == "mid_block":
                        vae_embed_0 = vae_embeds_0[:, -2]
                        vae_embed_1 = vae_embeds_1[:, -2]
                        vae_embed_2 = vae_embeds_2[:, -2]
                    else:
                        vae_embed_0 = vae_embeds_0[:, -1]
                        vae_embed_1 = vae_embeds_1[:, -1]
                        vae_embed_2 = vae_embeds_2[:, -1]

                    module.de_mod_0 = vae_embed_0.reshape(-1, self.lora_rank_vae, self.lora_rank_vae)
                    module.de_mod_1 = vae_embed_1.reshape(-1, self.lora_rank_vae, self.lora_rank_vae)
                    module.de_mod_2 = vae_embed_2.reshape(-1, self.lora_rank_vae, self.lora_rank_vae)

            for layer_name, module in self.unet.named_modules():
                if layer_name in self.unet_lora_layers:
                    split_name = layer_name.split(".")
                    if split_name[0] == "down_blocks":
                        block_id = int(split_name[1])
                        unet_embed_0 = unet_embeds_0[:, block_id]
                        unet_embed_1 = unet_embeds_1[:, block_id]
                        unet_embed_2 = unet_embeds_2[:, block_id]
                    elif split_name[0] == "mid_block":
                        unet_embed_0 = unet_embeds_0[:, 4]
                        unet_embed_1 = unet_embeds_1[:, 4]
                        unet_embed_2 = unet_embeds_2[:, 4]
                    elif split_name[0] == "up_blocks":
                        block_id = int(split_name[1]) + 5
                        unet_embed_0 = unet_embeds_0[:, block_id]
                        unet_embed_1 = unet_embeds_1[:, block_id]
                        unet_embed_2 = unet_embeds_2[:, block_id]
                    else:
                        unet_embed_0 = unet_embeds_0[:, -1]
                        unet_embed_1 = unet_embeds_1[:, -1]
                        unet_embed_2 = unet_embeds_2[:, -1]

                    module.de_mod_0 = unet_embed_0.reshape(-1, self.lora_rank_unet, self.lora_rank_unet)
                    module.de_mod_1 = unet_embed_1.reshape(-1, self.lora_rank_unet, self.lora_rank_unet)
                    module.de_mod_2 = unet_embed_2.reshape(-1, self.lora_rank_unet, self.lora_rank_unet)

        # ------------------------------------------
        # 3) adapter prior from c
        # ------------------------------------------
        if self.ablation_mode in ["no_eeg_all", "no_eeg_adapter", "vanilla_lora"]:
            prior = torch.zeros(
                B,
                self.prior_channels,
                self.prior_map_size,
                self.prior_map_size,
                device=c_t.device,
                dtype=c_t.dtype,
            )
        else:
            if c_adapter is None:
                raise RuntimeError("c_adapter should not be None when adapter prior is enabled.")
            prior = self._make_prior_from_c(c_adapter, B, c_t.device, c_t.dtype)

            if force_zero_prior or force_zero_c:
                prior = torch.zeros_like(prior)

        # ------------------------------------------
        # 4) one-step diffusion
        # ------------------------------------------
        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor

        # ------------------------------------------
        # Attention visualization switch
        # ------------------------------------------
        # 注意：
        # 1. 不要为了保存 attention 而关闭 down/up adapter，否则可视化对应的不是 full model。
        # 2. 这里只让 mid adapter 保存 cache，down/up 仍正常参与前向，但不保存大矩阵，避免 OOM。
        if hasattr(self.unet, "enable_adapter_attention_save"):
            self.unet.enable_adapter_attention_save(
                enabled=bool(return_attn_vis),
                stage="mid",
            )

        if return_attn_vis and hasattr(self.unet, "clear_adapter_attention"):
            self.unet.clear_adapter_attention()




        unet_out = self.unet(
            encoded_control,
            prior,
            self.timesteps,
            encoder_hidden_states=caption_enc,
        )

        model_pred = unet_out.sample



        # model_pred = self.unet(
        #     encoded_control,
        #     prior,
        #     self.timesteps,
        #     encoder_hidden_states=caption_enc,
        # ).sample

        

        for name in ("alphas_cumprod", "alphas", "betas", "final_alpha_cumprod"):
            value = getattr(self.sched, name, None)
            if torch.is_tensor(value) and value.device != model_pred.device:
                setattr(self.sched, name, value.to(model_pred.device))

        x_denoised = self.sched.step(
            model_pred,
            self.timesteps.to(model_pred.device),
            encoded_control,
            return_dict=True,
        ).prev_sample

        output_image = self.vae.decode(
            x_denoised / self.vae.config.scaling_factor
        ).sample.clamp(-1, 1)

        vis_data = None
        if return_attn_vis and hasattr(self.unet, "collect_adapter_attention"):
            vis_data = self.unet.collect_adapter_attention()





        # return output_image
        if return_attn_vis:
            return output_image, vis_data
        return output_image


    # ----------------------------------------
    # save
    # ----------------------------------------
    def save_model(self, outf):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["aligned_dim"] = self.aligned_dim
        sd["prior_channels"] = self.prior_channels
        sd["prior_map_size"] = self.prior_map_size
        sd["seg_in_channels"] = self.seg_in_channels

        sd["state_dict_unet"] = {
            k: v for k, v in self.unet.state_dict().items()
            if "adapter_blocks" in k or "conv_in" in k or "lora" in k
        }
        sd["state_dict_vae"] = {
            k: v for k, v in self.vae.state_dict().items()
            if "lora" in k or "skip_conv" in k
        }
        sd["state_dict_vae_de_mlp_1"] = {k: v for k, v in self.vae_de_mlp_1.state_dict().items()}
        sd["state_dict_vae_de_mlp_2"] = {k: v for k, v in self.vae_de_mlp_2.state_dict().items()}
        sd["state_dict_vae_block_mlp"] = {k: v for k, v in self.vae_block_mlp.state_dict().items()}
        sd["state_dict_vae_fuse_mlp_0"] = {k: v for k, v in self.vae_fuse_mlp_0.state_dict().items()}
        sd["state_dict_vae_fuse_mlp_1"] = {k: v for k, v in self.vae_fuse_mlp_1.state_dict().items()}
        sd["state_dict_vae_fuse_mlp_2"] = {k: v for k, v in self.vae_fuse_mlp_2.state_dict().items()}
        sd["state_dict_unet_processor"] = {k: v for k, v in self.unet_processor.state_dict().items()}
        sd["state_dict_wpenet"] = {k: v for k, v in self.wpenet.state_dict().items()}
        sd["state_dict_eeg_mlp"] = {k: v for k, v in self.eeg_mlp.state_dict().items()}
        sd["state_dict_prior_mapper"] = {k: v for k, v in self.prior_mapper.state_dict().items()}
        sd["state_dict_text_c_proj"] = {k: v for k, v in self.text_c_proj.state_dict().items()}
        sd["state_dict_depth_c_net"] = {k: v for k, v in self.depth_c_net.state_dict().items()}
        sd["state_dict_edge_c_net"] = {k: v for k, v in self.edge_c_net.state_dict().items()}
        sd["state_dict_seg_c_net"] = {k: v for k, v in self.seg_c_net.state_dict().items()}
        sd["w"] = self.W
        sd["state_embeddings"] = {
            "state_dict_vae_block": self.vae_block_embeddings.state_dict(),
        }
        torch.save(sd, outf)
