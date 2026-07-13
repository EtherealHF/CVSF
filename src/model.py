# ============================================
# File: src/model.py
# ============================================
import torch
import os
import requests
from tqdm import tqdm
from diffusers import DDPMScheduler
from typing import Any


def make_1step_sched(pretrained_path: str) -> DDPMScheduler:
    """Create the one-step diffusion scheduler."""
    noise_scheduler_1step = DDPMScheduler.from_pretrained(pretrained_path, subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    return noise_scheduler_1step


def my_lora_fwd(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    """LoRA forward pass with optional block-wise CVSF modulation."""
    self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)

    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)

    elif adapter_names is not None:
        result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)

    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)

    else:
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            x_cast = x.to(lora_A.weight.dtype)
            x_drop = dropout(x_cast)
            _tmp = lora_A(x_drop)

            # =====================================================
            #   "unet_skip_0/1/2"
            # =====================================================
            current_de_mod = None
            if active_adapter.startswith("vae_skip_") or active_adapter.startswith("unet_skip_"):
                adapter_index = active_adapter.split("_")[-1]  # "0" / "1" / "2"
                if hasattr(self, f"de_mod_{adapter_index}"):
                    current_de_mod = getattr(self, f"de_mod_{adapter_index}")

            if current_de_mod is not None:
                if isinstance(lora_A, torch.nn.Conv2d):
                    # _tmp: [B, r, H, W], de_mod: [B, r, r]
                    _tmp = torch.einsum("...khw,...kr->...rhw", _tmp, current_de_mod)
                elif isinstance(lora_A, torch.nn.Linear):
                    # _tmp: [B, L, r], de_mod: [B, r, r]
                    _tmp = torch.einsum("...lk,...kr->...lr", _tmp, current_de_mod)
                else:
                    raise NotImplementedError("my_lora_fwd currently only supports Conv2d and Linear for LoRA")

            result = result + lora_B(_tmp) * scaling

        result = result.to(torch_result_dtype)

    return result


def download_url(url: str, outf: str) -> None:
    """Download a file with a progress bar."""
    if not os.path.exists(outf):
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(outf, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            raise RuntimeError(f"Download incomplete: {url}")
