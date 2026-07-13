# ============================================
# File: src/inference_cvsf.py
# Supports:
#   - normal inference
#   - batch-wise mismatching ablation:
#       * shuffle_lora_in_batch
#       * shuffle_adapter_in_batch
#   - cognition condition ablations
#   - paper-style attention / response visualization:
#       output_dir/attn_vis/raw/
#       output_dir/attn_vis/input/
#       output_dir/attn_vis/overlay_input/
#       output_dir/attn_vis/attn_mean/
#       output_dir/attn_vis/attn_maxq/
#       output_dir/attn_vis/attn_topk/
#       output_dir/attn_vis/prior_energy/
#       output_dir/attn_vis/residual_energy/
#       output_dir/attn_vis/residual_absmax/
# ============================================

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import warnings
warnings.filterwarnings("ignore")

import gc
import csv
import random
import numpy as np
import torch
import torch.nn.functional as F
import transformers

from PIL import Image
import matplotlib.cm as cm

from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers.utils.import_utils import is_xformers_available

from de_net import DEResNet
from cvsf import CVSF
from my_utils.training_utils import parse_args_paired_training, ExposurePairedDataset
from my_utils.stage1_img_encoder import load_stage1_img_enc, Stage1ImageEncoderWrapper
from my_utils.image_encoder import ImageEncoder


def build_img_enc_fn(args):
    return ImageEncoder(
        backbone=args.stage1_backbone,
        out_dim=args.stage1_embed_dim,
        pretrained=False,
    )


def _pad_to_multiple(x, multiple=8):
    B, C, H, W = x.shape
    ph = (multiple - H % multiple) % multiple
    pw = (multiple - W % multiple) % multiple
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, ph, pw, (H, W)


def _pad_like_aux(aux, ph, pw):
    if aux is None:
        return None
    if ph or pw:
        aux = F.pad(aux, (0, pw, 0, ph), mode="reflect")
    return aux


def _make_hann_window(h, w, device, dtype):
    wy = torch.hann_window(h, periodic=False, device=device, dtype=dtype).view(h, 1)
    wx = torch.hann_window(w, periodic=False, device=device, dtype=dtype).view(1, w)
    return (wy @ wx).clamp_min(1e-6)


def _slice_vis_dict_one(vis_dict, bi):
    out = {}
    for k, v in vis_dict.items():
        if torch.is_tensor(v):
            out[k] = v[bi:bi + 1]
        else:
            out[k] = v
    return out


def _extract_2d_map(feat_map):
    """
    Convert tensor / ndarray to 2D numpy array.

    Supports:
        [1,1,H,W]
        [1,H,W]
        [H,W]
        [C,H,W]
        [1,C,H,W]
    """
    if torch.is_tensor(feat_map):
        x = feat_map.detach().float().cpu()

        if x.dim() == 4:
            if x.shape[1] > 1:
                x = x.pow(2).mean(dim=1, keepdim=True).sqrt()
            x = x[0, 0]
        elif x.dim() == 3:
            if x.shape[0] > 1:
                x = x.pow(2).mean(dim=0).sqrt()
            else:
                x = x[0]
        elif x.dim() == 2:
            pass
        else:
            raise ValueError(f"Unsupported feat_map shape: {tuple(x.shape)}")

        return x.numpy().astype(np.float32)

    arr = np.asarray(feat_map, dtype=np.float32)

    if arr.ndim == 4:
        arr = arr[0]
        if arr.shape[0] > 1:
            arr = np.sqrt(np.mean(arr ** 2, axis=0))
        else:
            arr = arr[0]
    elif arr.ndim == 3:
        if arr.shape[0] > 1:
            arr = np.sqrt(np.mean(arr ** 2, axis=0))
        else:
            arr = arr[0]
    elif arr.ndim == 2:
        pass
    else:
        raise ValueError(f"Unsupported feat_map ndarray shape: {arr.shape}")

    return arr.astype(np.float32)


def _normalize_map(
    arr,
    robust=True,
    robust_low=70,
    robust_high=99.5,
):
    """
    Normalize heatmap to [0,1].
    """
    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    if robust:
        lo = np.percentile(arr, robust_low)
        hi = np.percentile(arr, robust_high)
    else:
        lo = float(arr.min())
        hi = float(arr.max())

    if hi <= lo + 1e-12:
        return np.zeros_like(arr, dtype=np.float32)

    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo + 1e-6)

    return arr.astype(np.float32)

def _gaussian_kernel2d(kernel_size=5, sigma=1.0, device="cpu", dtype=torch.float32):
    """
    Create a 2D Gaussian kernel for smoothing resized attention maps.
    """
    if kernel_size <= 1:
        return None

    if kernel_size % 2 == 0:
        kernel_size += 1

    ax = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")

    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum().clamp_min(1e-12)

    return kernel.view(1, 1, kernel_size, kernel_size)


def _resize_map_tensor(
    arr_norm,
    target_hw,
    pixelated=False,
    smooth_kernel=5,
    smooth_sigma=1.0,
):
    """
    Resize low-resolution attention map to image resolution.

    pixelated=True:
        nearest interpolation, keeps original blocky structure.

    pixelated=False:
        bilinear interpolation + optional Gaussian smoothing,
        better for paper-style visualization.
    """
    x = torch.from_numpy(arr_norm).float()[None, None, :, :]

    if target_hw is not None:
        if pixelated:
            x = F.interpolate(
                x,
                size=target_hw,
                mode="nearest",
            )
        else:
            x = F.interpolate(
                x,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )

            if smooth_kernel is not None and smooth_kernel > 1:
                k = _gaussian_kernel2d(
                    kernel_size=int(smooth_kernel),
                    sigma=float(smooth_sigma),
                    device=x.device,
                    dtype=x.dtype,
                )
                pad = k.shape[-1] // 2
                x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
                x = F.conv2d(x, k)

    x = x[0, 0]

    x = x - x.min()
    x = x / x.max().clamp_min(1e-6)

    return x.numpy().astype(np.float32)



def _colorize_map(arr_norm, cmap_name="turbo"):
    try:
        cmap = cm.get_cmap(cmap_name)
    except Exception:
        cmap = cm.get_cmap("viridis")

    color = cmap(arr_norm)[..., :3]
    color = (color * 255).astype(np.uint8)

    return color

def _prepare_vis_map(
    feat_map,
    target_hw,
    pixelated=False,
    robust=True,
    robust_low=50,
    robust_high=99.5,
    heat_threshold=0.0,
    gamma=0.45,
    smooth_kernel=5,
    smooth_sigma=1.0,
    keep_top_percent=None,
):
    arr = _extract_2d_map(feat_map)

    arr_norm = _normalize_map(
        arr,
        robust=robust,
        robust_low=robust_low,
        robust_high=robust_high,
    )

    if heat_threshold > 0:
        arr_norm = np.maximum(arr_norm - float(heat_threshold), 0.0)
        arr_norm = arr_norm / (arr_norm.max() + 1e-6)

    if gamma is not None:
        arr_norm = np.power(arr_norm, float(gamma)).astype(np.float32)

    arr_norm = _resize_map_tensor(
        arr_norm,
        target_hw=target_hw,
        pixelated=pixelated,
        smooth_kernel=smooth_kernel,
        smooth_sigma=smooth_sigma,
    )

    # Keep only the strongest responses when requested.
    if keep_top_percent is not None and 0 < keep_top_percent < 100:
        thr = np.percentile(arr_norm, 100 - keep_top_percent)
        arr_norm = np.where(arr_norm >= thr, arr_norm, 0.0)
        arr_norm = arr_norm / (arr_norm.max() + 1e-6)

    return arr_norm



def _save_heatmap_png(
    feat_map,
    save_path,
    target_hw=None,
    cmap_name="turbo",
    pixelated=False,
    robust=True,
    robust_low=70,
    robust_high=99.5,
    heat_threshold=0.0,
    gamma=0.65,
    smooth_kernel=5,
    smooth_sigma=1.0,
):
    """
    Save standalone heatmap.

    smooth version:
        pixelated=False

    blocky debug version:
        pixelated=True
    """
    if feat_map is None:
        return

    arr_norm = _prepare_vis_map(
        feat_map,
        target_hw=target_hw,
        pixelated=pixelated,
        robust=robust,
        robust_low=robust_low,
        robust_high=robust_high,
        heat_threshold=heat_threshold,
        gamma=gamma,
        smooth_kernel=0 if pixelated else smooth_kernel,
        smooth_sigma=smooth_sigma,
    )

    color = _colorize_map(arr_norm, cmap_name=cmap_name)
    Image.fromarray(color).save(save_path)



def _overlay_heatmap_on_image(
    img_tensor,
    feat_map,
    save_path,
    alpha=1.0,
    cmap_name="turbo",
    robust=True,
    robust_low=50,
    robust_high=99.5,
    heat_threshold=0.0,
    gamma=0.45,
    smooth_kernel=7,
    smooth_sigma=1.4,
):
    """
    Save heatmap overlay on image.

    Stronger-spot version:
    - lower threshold
    - stronger mask
    - brighter hot regions
    """
    if feat_map is None:
        return

    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    base = (img_tensor.detach().float().cpu() * 0.5 + 0.5).clamp(0, 1)
    H, W = base.shape[-2:]

    arr_norm = _prepare_vis_map(
        feat_map,
        target_hw=(H, W),
        pixelated=False,
        robust=robust,
        robust_low=robust_low,
        robust_high=robust_high,
        heat_threshold=heat_threshold,
        gamma=gamma,
        smooth_kernel=smooth_kernel,
        smooth_sigma=smooth_sigma,
    )

    arr_norm = arr_norm - arr_norm.min()
    arr_norm = arr_norm / (arr_norm.max() + 1e-6)

    mask_np = np.power(arr_norm, 0.5).astype(np.float32)

    color_np = _colorize_map(arr_norm, cmap_name=cmap_name).astype(np.float32) / 255.0
    color = torch.from_numpy(color_np).permute(2, 0, 1).float()
    mask = torch.from_numpy(mask_np).float()[None, :, :].clamp(0, 1)

    # 寮哄寲浜偣鍖哄煙鐨勫彔鍔?    overlay = base[0] * (1.0 - alpha * mask) + color * (alpha * mask)
    overlay = overlay.clamp(0, 1)

    # 棰濆澧炲己鐑偣鐨勪寒搴︼紙鍙€変絾寰堟湁鏁堬級
    overlay = torch.where(
        mask > 0.35,
        torch.clamp(overlay * 1.12, 0, 1),
        overlay
    )

    to_pil = transforms.ToPILImage()
    to_pil(overlay).save(save_path)




def _attn_to_spatial_map(attn_softmax, h, w):
    """
    Backward-compatible old-format attention matrix to spatial map.

    attn_softmax:
        [1, N, N] or [N, N]
    return:
        [1, 1, h, w]
    """
    if attn_softmax.dim() == 2:
        attn_softmax = attn_softmax.unsqueeze(0)

    attn_map = attn_softmax.float().mean(dim=1)

    N = attn_map.shape[-1]
    if N != h * w:
        side = int(N ** 0.5)
        if side * side == N:
            h, w = side, side
        else:
            raise ValueError(f"Cannot reshape attention map: N={N}, target h*w={h*w}")

    attn_map = attn_map.view(1, 1, h, w)

    attn_min = attn_map.amin(dim=(-2, -1), keepdim=True)
    attn_max = attn_map.amax(dim=(-2, -1), keepdim=True)
    attn_map = (attn_map - attn_min) / (attn_max - attn_min).clamp_min(1e-6)

    return attn_map



@torch.no_grad()
def _estimate_deg_score(x_in, net_de, args, noise_gen=None):
    repeat_n = max(1, int(getattr(args, "deg_repeat_times", 1)))
    input_noise_std = float(getattr(args, "deg_input_noise_std", 0.0))
    score_noise_std = float(getattr(args, "deg_score_noise_std", 0.0))
    fuse_mode = getattr(args, "deg_fuse_mode", "mean")
    override_mode = getattr(args, "deg_override_mode", "none")

    deg_list = []

    for _ in range(repeat_n):
        x_noisy = x_in

        if input_noise_std > 0:
            eps = torch.randn(
                x_in.shape,
                generator=noise_gen,
                device=x_in.device,
                dtype=x_in.dtype,
            )
            x_noisy = (x_in + eps * input_noise_std).clamp(-1, 1)

        deg = net_de(x_noisy)

        if score_noise_std > 0:
            eps_deg = torch.randn(
                deg.shape,
                generator=noise_gen,
                device=deg.device,
                dtype=deg.dtype,
            )
            deg = deg + eps_deg * score_noise_std

        deg_list.append(deg)

    deg_stack = torch.stack(deg_list, dim=0)

    if fuse_mode == "first":
        deg_fused = deg_stack[0]
    elif fuse_mode == "median":
        deg_fused = deg_stack.median(dim=0).values
    else:
        deg_fused = deg_stack.mean(dim=0)

    if override_mode == "zero":
        deg_fused = torch.zeros_like(deg_fused)
    elif override_mode == "random":
        deg_fused = torch.rand(
            deg_fused.shape,
            generator=noise_gen,
            device=deg_fused.device,
            dtype=deg_fused.dtype,
        )

    deg_fused = deg_fused.clamp(0, 1)

    std_scalar = 0.0
    if deg_stack.shape[0] > 1:
        std_scalar = float(deg_stack.std(dim=0).mean().item())

    stat = {
        "repeat_std": std_scalar,
        "fused_mean": float(deg_fused.mean().item()),
        "fused_min": float(deg_fused.min().item()),
        "fused_max": float(deg_fused.max().item()),
        "deg_values": deg_fused.detach().mean(dim=0).float().cpu().tolist(),
    }

    return deg_fused, stat


def _forward_cvsf(
    net_sr,
    x,
    deg,
    prompt,
    args,
    depth_map=None,
    edge_map=None,
    seg_map=None,
    return_attn_vis=False,
):
    return net_sr(
        x,
        deg,
        prompt,

        shuffle_lora_in_batch=getattr(args, "shuffle_lora_in_batch", False),
        shuffle_adapter_in_batch=getattr(args, "shuffle_adapter_in_batch", False),
        shuffle_seed=getattr(args, "shuffle_seed", None),

        force_zero_prior=getattr(args, "force_zero_prior", False),
        force_zero_c=getattr(args, "force_zero_c", False),
        force_noise_c=getattr(args, "force_noise_c", False),
        noise_c_scale=getattr(args, "noise_c_scale", 1.0),
        noise_c_seed=getattr(args, "noise_c_seed", None),
        noise_c_match_stats=getattr(args, "noise_c_match_stats", False),

        force_text_c=getattr(args, "force_text_c", False),
        text_c_pool=getattr(args, "text_c_pool", "mean"),
        text_c_detach=getattr(args, "text_c_detach", False),

        c_guidance_type=getattr(args, "c_guidance_type", "image"),
        depth_map=depth_map,
        edge_map=edge_map,
        seg_map=seg_map,

        force_band_noise_c=getattr(args, "force_band_noise_c", False),
        band_noise_c_scale=getattr(args, "band_noise_c_scale", 1.0),
        band_noise_c_seed=getattr(args, "band_noise_c_seed", None),
        band_start_ratio=getattr(args, "band_start_ratio", 0.2),
        band_end_ratio=getattr(args, "band_end_ratio", 0.5),
        band_noise_match_stats=getattr(args, "band_noise_match_stats", False),

        return_attn_vis=return_attn_vis,
    )


@torch.no_grad()
def tiled_infer_full_image(
    x_src,
    net_sr,
    net_de,
    prompt,
    tile,
    overlap,
    device,
    dtype,
    args,
    noise_gen=None,
    depth_map=None,
    edge_map=None,
    seg_map=None,
    return_vis=False,
):
    x_src = x_src.to(device=device, dtype=dtype)
    x_src, ph, pw, (H0, W0) = _pad_to_multiple(x_src, multiple=8)

    depth_map = None if depth_map is None else depth_map.to(device=device, dtype=dtype)
    edge_map = None if edge_map is None else edge_map.to(device=device, dtype=dtype)
    seg_map = None if seg_map is None else seg_map.to(device=device, dtype=dtype)

    depth_map = _pad_like_aux(depth_map, ph, pw)
    edge_map = _pad_like_aux(edge_map, ph, pw)
    seg_map = _pad_like_aux(seg_map, ph, pw)

    B, _, H, W = x_src.shape

    # Tiled visualization saves center tile only.
    save_center_tile_vis = bool(return_vis and tile > 0 and tile < max(H, W))
    center_tile_vis_data = None

    # =========================
    # Non-tiled full-image mode
    # =========================
    if tile == 0 or tile >= max(H, W):
        deg, stat = _estimate_deg_score(
            x_src,
            net_de,
            args,
            noise_gen=noise_gen,
        )

        if return_vis:
            out, vis_data = _forward_cvsf(
                net_sr=net_sr,
                x=x_src,
                deg=deg,
                prompt=prompt,
                args=args,
                depth_map=depth_map,
                edge_map=edge_map,
                seg_map=seg_map,
                return_attn_vis=True,
            )
        else:
            out = _forward_cvsf(
                net_sr=net_sr,
                x=x_src,
                deg=deg,
                prompt=prompt,
                args=args,
                depth_map=depth_map,
                edge_map=edge_map,
                seg_map=seg_map,
                return_attn_vis=False,
            )
            vis_data = None

        if ph or pw:
            out = out[:, :, :H0, :W0]

        return out, [stat], vis_data

    # =========================
    # Tiled mode
    # =========================
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("tile_size must be larger than tile_overlap")

    win = _make_hann_window(tile, tile, device, dtype)[None, None, :, :]
    deg_stats = []

    ys = list(range(0, max(H - tile + 1, 1), stride))
    xs = list(range(0, max(W - tile + 1, 1), stride))

    if len(ys) == 0:
        ys = [0]
    if len(xs) == 0:
        xs = [0]

    if ys[-1] != H - tile:
        ys.append(max(H - tile, 0))
    if xs[-1] != W - tile:
        xs.append(max(W - tile, 0))

    if save_center_tile_vis:
        center_y = H // 2
        center_x = W // 2
        best_top, best_left = min(
            [(yy, xx) for yy in ys for xx in xs],
            key=lambda p: abs((p[0] + tile // 2) - center_y) + abs((p[1] + tile // 2) - center_x),
        )
    else:
        best_top, best_left = None, None

    total_tiles = len(ys) * len(xs)
    pbar = tqdm(total=total_tiles, desc="Processing Tiles", leave=False)

    acc = None
    wei = None

    for top in ys:
        for left in xs:
            xs_crop = x_src[:, :, top:top + tile, left:left + tile]

            depth_crop = None if depth_map is None else depth_map[:, :, top:top + tile, left:left + tile]
            edge_crop = None if edge_map is None else edge_map[:, :, top:top + tile, left:left + tile]
            seg_crop = None if seg_map is None else seg_map[:, :, top:top + tile, left:left + tile]

            deg, stat = _estimate_deg_score(
                xs_crop,
                net_de,
                args,
                noise_gen=noise_gen,
            )
            deg_stats.append(stat)

            need_tile_vis = bool(save_center_tile_vis and top == best_top and left == best_left)

            if need_tile_vis:
                pred, tile_vis_data = _forward_cvsf(
                    net_sr=net_sr,
                    x=xs_crop,
                    deg=deg,
                    prompt=prompt,
                    args=args,
                    depth_map=depth_crop,
                    edge_map=edge_crop,
                    seg_map=seg_crop,
                    return_attn_vis=True,
                )
                center_tile_vis_data = {
                    "vis_data": tile_vis_data,
                    "top": top,
                    "left": left,
                    "tile": tile,
                }
            else:
                pred = _forward_cvsf(
                    net_sr=net_sr,
                    x=xs_crop,
                    deg=deg,
                    prompt=prompt,
                    args=args,
                    depth_map=depth_crop,
                    edge_map=edge_crop,
                    seg_map=seg_crop,
                    return_attn_vis=False,
                )

            if acc is None:
                acc = torch.zeros(
                    (B, pred.shape[1], H, W),
                    dtype=dtype,
                    device=device,
                )
                wei = torch.zeros(
                    (B, 1, H, W),
                    dtype=dtype,
                    device=device,
                )

            acc[:, :, top:top + tile, left:left + tile] += pred * win
            wei[:, :, top:top + tile, left:left + tile] += win

            pbar.update(1)

    pbar.close()

    out = acc / wei.clamp_min(1e-6)

    if ph or pw:
        out = out[:, :, :H0, :W0]

    return out, deg_stats, center_tile_vis_data

def _get_attention_save_keys(args):
    """
    Default keys for paper-style visualization.

    You can override it by:
        --attention_save_keys attn_topk_excess,attn_topk,residual_energy,prior_energy
    """
    default_keys = [
        # Recommended main-figure maps
        "attn_topk_excess",
        "attn_topk",
        "residual_energy",
        "prior_energy",

        # Optional analysis maps
        "attn_mean_excess",
        "attn_maxq_excess",
        "attn_mean",
        "attn_maxq",
        "qk_logits_max",
        "qk_logits_pos",
        "residual_absmax",
    ]

    s = getattr(args, "attention_save_keys", None)

    if s is None or str(s).strip() == "":
        return default_keys

    return [x.strip() for x in str(s).split(",") if x.strip()]



def _save_attention_outputs(
    *,
    args,
    accelerator,
    vis_data,
    batch_val,
    x_src,
    x_pred,
    x_tgt,
    step,
    B,
    to_pil,
    attention_save_dirname,
    saved_attention_count,
    attention_save_limit,
):
    """
    Save attention / response visualization.

    Main output:
        output_dir/attn_vis/overlay_pred/<map_name>/*_mid_overlay_pred.png

    Recommended paper figures:
        overlay_pred/attn_topk_excess/
        overlay_pred/residual_energy/

    Four interpretation points:
        prior_energy       -> before QKV / prior-fused spatial energy
        qk_logits_max      -> QK^T / sqrt(dk), before softmax
        attn_topk_excess   -> softmax attention map, highlighted
        residual_energy    -> final residual response
    """
    attn_root = os.path.join(args.output_dir, attention_save_dirname)

    attn_raw_dir = os.path.join(attn_root, "raw")
    attn_input_dir = os.path.join(attn_root, "input")
    attn_pred_dir = os.path.join(attn_root, "pred")
    attn_gt_dir = os.path.join(attn_root, "gt")

    overlay_input_root = os.path.join(attn_root, "overlay_input")
    overlay_pred_root = os.path.join(attn_root, "overlay_pred")
    overlay_gt_root = os.path.join(attn_root, "overlay_gt")

    for d in [
        attn_raw_dir,
        attn_input_dir,
        attn_pred_dir,
        attn_gt_dir,
        overlay_input_root,
        overlay_pred_root,
        overlay_gt_root,
    ]:
        os.makedirs(d, exist_ok=True)

    is_tile_vis = isinstance(vis_data, dict) and ("vis_data" in vis_data)

    if is_tile_vis:
        real_vis_data = vis_data["vis_data"]
        tile_top = int(vis_data["top"])
        tile_left = int(vis_data["left"])
        tile_size = int(vis_data["tile"])
    else:
        real_vis_data = vis_data
        tile_top = 0
        tile_left = 0
        tile_size = None

    if real_vis_data is None:
        accelerator.print("[Attention] real_vis_data is None.")
        return saved_attention_count

    mid_vis = real_vis_data.get("mid", {})

    if mid_vis is None or len(mid_vis) == 0:
        accelerator.print("[Attention] vis_data['mid'] is empty. Check whether mid adapter is enabled.")
        return saved_attention_count

    if "lq_path" in batch_val:
        paths = batch_val["lq_path"]
    elif "lq_paths" in batch_val:
        paths = batch_val["lq_paths"]
    else:
        paths = [f"val_{step:06d}_{i}" for i in range(B)]

    save_keys = _get_attention_save_keys(args)

    # Visualization style.
    # These defaults are tuned to avoid blocky / all-purple maps.
    alpha = float(getattr(args, "attn_overlay_alpha", 1.0))
    cmap_name = str(getattr(args, "attn_cmap", "turbo"))

    robust_low = float(getattr(args, "attn_robust_low", 50))
    robust_high = float(getattr(args, "attn_robust_high", 99.5))

    heat_threshold = float(getattr(args, "attn_heat_threshold", 0.02))
    gamma = float(getattr(args, "attn_gamma", 0.45))

    smooth_kernel = int(getattr(args, "attn_smooth_kernel", 7))
    smooth_sigma = float(getattr(args, "attn_smooth_sigma", 1.4))

    save_blocky = bool(getattr(args, "save_blocky_attention", False))
    save_overlay_input = bool(getattr(args, "save_overlay_input", False))
    save_overlay_gt = bool(getattr(args, "save_overlay_gt", False))

    for bi in range(B):
        if saved_attention_count >= attention_save_limit:
            break

        stem_i = os.path.splitext(os.path.basename(str(paths[bi])))[0]

        if is_tile_vis:
            stem_i = f"{stem_i}_tile_y{tile_top}_x{tile_left}"

        one_vis = _slice_vis_dict_one(mid_vis, bi)

        raw_save_path = os.path.join(attn_raw_dir, f"{stem_i}_mid.pt")
        torch.save(one_vis, raw_save_path)

        maps_to_save = {}

        # New-format keys
        for key in save_keys:
            if key in one_vis:
                maps_to_save[key] = one_vis[key]

        # Old-version compatibility
        if "attn_map" in one_vis and "attn_map" not in maps_to_save:
            maps_to_save["attn_map"] = one_vis["attn_map"]

        elif "attn_softmax" in one_vis and "attn_map" not in maps_to_save:
            if "pre_kv_feature" in one_vis:
                h, w = one_vis["pre_kv_feature"].shape[-2:]
            elif "prior" in one_vis:
                h, w = one_vis["prior"].shape[-2:]
            else:
                N = one_vis["attn_softmax"].shape[-1]
                side = int(N ** 0.5)
                h, w = side, side

            maps_to_save["attn_map"] = _attn_to_spatial_map(
                one_vis["attn_softmax"],
                h,
                w,
            )

        if "pre_kv_feature_mean" in one_vis:
            maps_to_save.setdefault(
                "pre_kv_mean",
                one_vis["pre_kv_feature_mean"],
            )

        elif "pre_kv_feature" in one_vis:
            maps_to_save.setdefault(
                "pre_kv_energy",
                one_vis["pre_kv_feature"]
                .detach()
                .float()
                .pow(2)
                .mean(dim=1, keepdim=True)
                .sqrt(),
            )

        if "residual_mean" in one_vis:
            maps_to_save.setdefault(
                "residual_mean",
                one_vis["residual_mean"],
            )

        elif "residual_branch" in one_vis:
            maps_to_save.setdefault(
                "residual_energy_old",
                one_vis["residual_branch"]
                .detach()
                .float()
                .pow(2)
                .mean(dim=1, keepdim=True)
                .sqrt(),
            )

        if len(maps_to_save) == 0:
            accelerator.print(
                f"[Attention] No valid maps found for {stem_i}. "
                f"Keys={list(one_vis.keys())}"
            )
            continue

        # If tiled mode, only save center tile.
        if is_tile_vis:
            x_src_vis = x_src[
                bi:bi + 1,
                :,
                tile_top:tile_top + tile_size,
                tile_left:tile_left + tile_size,
            ]

            x_pred_vis = x_pred[
                bi:bi + 1,
                :,
                tile_top:tile_top + tile_size,
                tile_left:tile_left + tile_size,
            ]

            if x_tgt is not None:
                x_tgt_vis = x_tgt[
                    bi:bi + 1,
                    :,
                    tile_top:tile_top + tile_size,
                    tile_left:tile_left + tile_size,
                ]
            else:
                x_tgt_vis = None

        else:
            x_src_vis = x_src[bi:bi + 1]
            x_pred_vis = x_pred[bi:bi + 1]
            x_tgt_vis = None if x_tgt is None else x_tgt[bi:bi + 1]

        target_hw = tuple(x_pred_vis.shape[-2:])

        # Save original input / pred / gt for comparison.
        x_src_img = (x_src_vis.detach().float().cpu() * 0.5 + 0.5).clamp(0, 1)
        to_pil(x_src_img[0]).save(os.path.join(attn_input_dir, f"{stem_i}.png"))

        x_pred_img = (x_pred_vis.detach().float().cpu() * 0.5 + 0.5).clamp(0, 1)
        to_pil(x_pred_img[0]).save(os.path.join(attn_pred_dir, f"{stem_i}.png"))

        if x_tgt_vis is not None:
            x_tgt_img = (x_tgt_vis.detach().float().cpu() * 0.5 + 0.5).clamp(0, 1)
            to_pil(x_tgt_img[0]).save(os.path.join(attn_gt_dir, f"{stem_i}.png"))

        # Debug stats
        for map_name, map_tensor in maps_to_save.items():
            try:
                arr_dbg = _extract_2d_map(map_tensor)
                accelerator.print(
                    f"[AttentionDebug] {stem_i} | {map_name} | "
                    f"shape={arr_dbg.shape}, "
                    f"min={arr_dbg.min():.6e}, "
                    f"max={arr_dbg.max():.6e}, "
                    f"mean={arr_dbg.mean():.6e}, "
                    f"std={arr_dbg.std():.6e}"
                )
            except Exception as e:
                accelerator.print(
                    f"[AttentionDebug] failed on {stem_i} | {map_name}: {e}"
                )

        # Save heatmap and overlay.
        for map_name, map_tensor in maps_to_save.items():
            map_dir = os.path.join(attn_root, map_name)
            os.makedirs(map_dir, exist_ok=True)

            # Smooth standalone heatmap.
            _save_heatmap_png(
                map_tensor,
                os.path.join(map_dir, f"{stem_i}_mid_smooth.png"),
                target_hw=target_hw,
                cmap_name=cmap_name,
                pixelated=False,
                robust=True,
                robust_low=robust_low,
                robust_high=robust_high,
                heat_threshold=0.0,
                gamma=gamma,
                smooth_kernel=smooth_kernel,
                smooth_sigma=smooth_sigma,
            )

            # Optional blocky heatmap for debugging original grid.
            if save_blocky:
                _save_heatmap_png(
                    map_tensor,
                    os.path.join(map_dir, f"{stem_i}_mid_blocky.png"),
                    target_hw=target_hw,
                    cmap_name=cmap_name,
                    pixelated=True,
                    robust=True,
                    robust_low=robust_low,
                    robust_high=robust_high,
                    heat_threshold=0.0,
                    gamma=gamma,
                    smooth_kernel=0,
                    smooth_sigma=smooth_sigma,
                )

            # Main output: overlay on pred.
            overlay_pred_dir = os.path.join(overlay_pred_root, map_name)
            os.makedirs(overlay_pred_dir, exist_ok=True)

            _overlay_heatmap_on_image(
                x_pred_vis,
                map_tensor,
                os.path.join(
                    overlay_pred_dir,
                    f"{stem_i}_mid_overlay_pred.png",
                ),
                alpha=alpha,
                cmap_name=cmap_name,
                robust=True,
                robust_low=robust_low,
                robust_high=robust_high,
                heat_threshold=heat_threshold,
                gamma=gamma,
                smooth_kernel=smooth_kernel,
                smooth_sigma=smooth_sigma,
            )

            # Optional: overlay on input.
            if save_overlay_input:
                overlay_input_dir = os.path.join(overlay_input_root, map_name)
                os.makedirs(overlay_input_dir, exist_ok=True)

                _overlay_heatmap_on_image(
                    x_src_vis,
                    map_tensor,
                    os.path.join(
                        overlay_input_dir,
                        f"{stem_i}_mid_overlay_input.png",
                    ),
                    alpha=alpha,
                    cmap_name=cmap_name,
                    robust=True,
                    robust_low=robust_low,
                    robust_high=robust_high,
                    heat_threshold=heat_threshold,
                    gamma=gamma,
                    smooth_kernel=smooth_kernel,
                    smooth_sigma=smooth_sigma,
                )

            # Optional: overlay on gt.
            if save_overlay_gt and x_tgt_vis is not None:
                overlay_gt_dir = os.path.join(overlay_gt_root, map_name)
                os.makedirs(overlay_gt_dir, exist_ok=True)

                _overlay_heatmap_on_image(
                    x_tgt_vis,
                    map_tensor,
                    os.path.join(
                        overlay_gt_dir,
                        f"{stem_i}_mid_overlay_gt.png",
                    ),
                    alpha=alpha,
                    cmap_name=cmap_name,
                    robust=True,
                    robust_low=robust_low,
                    robust_high=robust_high,
                    heat_threshold=heat_threshold,
                    gamma=gamma,
                    smooth_kernel=smooth_kernel,
                    smooth_sigma=smooth_sigma,
                )

        saved_attention_count += 1

    accelerator.print(
        f"[Attention] saved {saved_attention_count}/{attention_save_limit} attention/response maps."
    )

    return saved_attention_count



def main(args):
    config = OmegaConf.load(args.base_config)
    if args.input_meta is not None:
        config.validation["meta_info"] = args.input_meta

    if args.sd_path is None:
        from huggingface_hub import snapshot_download
        sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    else:
        sd_path = args.sd_path

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.deg_repeat_times < 1:
        raise ValueError("--deg_repeat_times must be >= 1")

    if args.tile_size > 0 and args.tile_overlap >= args.tile_size:
        raise ValueError("--tile_overlap must be smaller than --tile_size when tile_size > 0")

    save_attention_vis = bool(getattr(args, "save_attention_vis", False))
    attention_save_limit = int(getattr(args, "attention_save_limit", 4))
    attention_save_stage = str(getattr(args, "attention_save_stage", "mid"))
    attention_save_dirname = str(getattr(args, "attention_save_dirname", "attn_vis"))

    if save_attention_vis and attention_save_stage != "mid":
        raise NotImplementedError("This version only supports attention_save_stage='mid'.")

    if save_attention_vis and int(args.tile_size) > 0:
        accelerator.print(
            "[Attention] save_attention_vis=True with tiled inference. "
            "Center tile attention will be saved to avoid OOM."
        )

    if save_attention_vis and int(args.tile_size) == 0:
        accelerator.print(
            "[Attention] save_attention_vis=True with full-image inference. "
            "This is recommended for paper visualization if memory allows."
        )

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "pred"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "gt"), exist_ok=True)

        if save_attention_vis:
            attn_root = os.path.join(args.output_dir, attention_save_dirname)
            os.makedirs(os.path.join(attn_root, "raw"), exist_ok=True)
            os.makedirs(os.path.join(attn_root, "input"), exist_ok=True)
            os.makedirs(os.path.join(attn_root, "overlay_input"), exist_ok=True)

    if not hasattr(args, "stage1_img_encoder_ckpt") or args.stage1_img_encoder_ckpt is None:
        raise ValueError("Please provide --stage1_img_encoder_ckpt for inference.")

    accelerator.print(f"[Stage1] loading image encoder from: {args.stage1_img_encoder_ckpt}")

    raw_img_enc = load_stage1_img_enc(
        build_img_enc_fn=lambda: build_img_enc_fn(args),
        ckpt_path=args.stage1_img_encoder_ckpt,
        device="cpu",
        strict=True,
    )

    stage1_img_enc = Stage1ImageEncoderWrapper(
        img_enc=raw_img_enc,
        image_size=args.stage1_input_size,
        freeze=True,
    )

    net_de = DEResNet(num_in_ch=3, num_degradation=2).cuda().eval()
    net_de.load_model(args.de_net_path)

    net_sr = CVSF(
        lora_rank_unet=args.lora_rank_unet,
        lora_rank_vae=args.lora_rank_vae,
        sd_path=sd_path,
        pretrained_path=args.pretrained_path,
        ablation_mode=args.ablation_mode,
        lora_zero_parts=args.lora_zero_parts,

        stage1_img_enc=stage1_img_enc,
        freeze_stage1_img_enc=True,
        aligned_dim=args.stage1_embed_dim,

        prior_map_size=args.prior_map_size,
        prior_channels=args.prior_channels,

        seg_in_channels=getattr(args, "seg_in_channels", 1),
    )

    net_sr.set_eval()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_sr.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if hasattr(config, "validation"):
        config.validation["infer_full_image"] = bool(
            config.validation.get("infer_full_image", True) or args.infer_full_image
        )

    dataset_val = ExposurePairedDataset(config.validation)
    dl_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=getattr(args, "infer_batch_size", 1),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    net_sr, dl_val = accelerator.prepare(net_sr, dl_val)
    net_de = accelerator.prepare(net_de)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    net_sr.to(accelerator.device, dtype=weight_dtype)
    net_de.to(accelerator.device, dtype=weight_dtype)

    to_pil = transforms.ToPILImage()

    deg_noise_gen = torch.Generator(device=accelerator.device)
    base_seed = int(args.seed) if args.seed is not None else torch.seed()
    deg_noise_gen.manual_seed(base_seed + 20260407)

    deg_stat_counter = 0
    deg_std_sum = 0.0
    deg_mean_sum = 0.0
    per_image_deg_rows = []

    saved_attention_count = 0

    for step, batch_val in enumerate(dl_val):
        x_src = batch_val["lq"].to(accelerator.device, non_blocking=True)

        x_tgt = batch_val.get("gt", None)
        if x_tgt is not None:
            x_tgt = x_tgt.to(accelerator.device, non_blocking=True)

        depth_map = batch_val.get("depth", None)
        edge_map = batch_val.get("edge", None)
        seg_map = batch_val.get("seg", None)

        if depth_map is not None:
            depth_map = depth_map.to(accelerator.device, non_blocking=True)
        if edge_map is not None:
            edge_map = edge_map.to(accelerator.device, non_blocking=True)
        if seg_map is not None:
            seg_map = seg_map.to(accelerator.device, non_blocking=True)

        B = x_src.shape[0]

        with torch.no_grad():
            pos_tag_prompt = [args.pos_prompt for _ in range(B)]

            need_vis_this_batch = (
                save_attention_vis
                and accelerator.is_main_process
                and saved_attention_count < attention_save_limit
            )

            x_pred, tile_deg_stats, vis_data = tiled_infer_full_image(
                x_src=x_src.detach(),
                net_sr=accelerator.unwrap_model(net_sr),
                net_de=accelerator.unwrap_model(net_de),
                prompt=pos_tag_prompt,
                tile=int(args.tile_size),
                overlap=int(args.tile_overlap),
                device=accelerator.device,
                dtype=weight_dtype,
                args=args,
                noise_gen=deg_noise_gen,
                depth_map=depth_map,
                edge_map=edge_map,
                seg_map=seg_map,
                return_vis=need_vis_this_batch,
            )

            for one_stat in tile_deg_stats:
                deg_stat_counter += 1
                deg_std_sum += float(one_stat["repeat_std"])
                deg_mean_sum += float(one_stat["fused_mean"])

            if len(tile_deg_stats) > 0:
                tile_repeat_std = [float(s["repeat_std"]) for s in tile_deg_stats]
                tile_fused_mean = [float(s["fused_mean"]) for s in tile_deg_stats]
                tile_fused_min = [float(s["fused_min"]) for s in tile_deg_stats]
                tile_fused_max = [float(s["fused_max"]) for s in tile_deg_stats]
                deg_dim_count = len(tile_deg_stats[0].get("deg_values", []))

                row = {
                    "batch_step": step,
                    "num_tiles": len(tile_deg_stats),
                    "tile_repeat_std_mean": sum(tile_repeat_std) / len(tile_repeat_std),
                    "tile_fused_mean_mean": sum(tile_fused_mean) / len(tile_fused_mean),
                    "tile_fused_min_mean": sum(tile_fused_min) / len(tile_fused_min),
                    "tile_fused_max_mean": sum(tile_fused_max) / len(tile_fused_max),
                }

                for di in range(deg_dim_count):
                    vals = [float(s["deg_values"][di]) for s in tile_deg_stats]
                    row[f"deg{di}"] = sum(vals) / len(vals)

                per_image_deg_rows.append(row)

            x_pred_v = (x_pred.cpu().detach() * 0.5 + 0.5).clamp(0, 1)

            if accelerator.is_main_process:
                pred_dir = os.path.join(args.output_dir, "pred")
                os.makedirs(pred_dir, exist_ok=True)

                if "lq_path" in batch_val:
                    paths = batch_val["lq_path"]
                elif "lq_paths" in batch_val:
                    paths = batch_val["lq_paths"]
                else:
                    paths = [f"val_{step:06d}_{i}" for i in range(B)]

                for bi in range(B):
                    stem_i = os.path.splitext(os.path.basename(str(paths[bi])))[0]
                    to_pil(x_pred_v[bi]).save(os.path.join(pred_dir, f"{stem_i}.png"))

            if x_tgt is not None:
                x_tgt_v = (x_tgt.cpu().detach() * 0.5 + 0.5).clamp(0, 1)

                if accelerator.is_main_process:
                    gt_dir = os.path.join(args.output_dir, "gt")
                    os.makedirs(gt_dir, exist_ok=True)

                    if "lq_path" in batch_val:
                        paths = batch_val["lq_path"]
                    elif "lq_paths" in batch_val:
                        paths = batch_val["lq_paths"]
                    else:
                        paths = [f"val_{step:06d}_{i}" for i in range(B)]

                    for bi in range(B):
                        stem_i = os.path.splitext(os.path.basename(str(paths[bi])))[0]
                        to_pil(x_tgt_v[bi]).save(os.path.join(gt_dir, f"{stem_i}.png"))

            if accelerator.is_main_process and need_vis_this_batch and vis_data is not None:
                saved_attention_count = _save_attention_outputs(
                    args=args,
                    accelerator=accelerator,
                    vis_data=vis_data,
                    batch_val=batch_val,
                    x_src=x_src,
                    x_pred=x_pred,
                    x_tgt=x_tgt,
                    step=step,
                    B=B,
                    to_pil=to_pil,
                    attention_save_dirname=attention_save_dirname,
                    saved_attention_count=saved_attention_count,
                    attention_save_limit=attention_save_limit,
                )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if accelerator.is_main_process and getattr(args, "save_deg_scores", False):
        if len(per_image_deg_rows) > 0:
            csv_name = getattr(args, "deg_score_csv_name", "deg_scores.csv")
            csv_path = os.path.join(args.output_dir, csv_name)

            deg_keys = sorted(
                [k for k in per_image_deg_rows[0].keys() if k.startswith("deg")]
            )

            fields = [
                "batch_step",
                "num_tiles",
                "tile_repeat_std_mean",
                "tile_fused_mean_mean",
                "tile_fused_min_mean",
                "tile_fused_max_mean",
            ] + deg_keys

            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(per_image_deg_rows)


if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
