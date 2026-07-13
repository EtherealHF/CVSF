# -*- coding: utf-8 -*-
# 评测脚本：按文件名末尾 -3.0/-4.0/-5.0 分组分别评测；FR 对齐仅支持 resize 或 skip
import os
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import sys
import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyiqa  # 需要 pyiqa
import tempfile
import shutil
from datetime import datetime

# -------------------------
# 环境与路径
# -------------------------
# 建议 CUDA_VISIBLE_DEVICES 在 torch 初始化前设置（你也可以删掉这一行，用外部环境变量控制）
os.environ.setdefault('CUDA_VISIBLE_DEVICES', "1,2")

# === 将项目根目录加入 sys.path： .../CVSF/ ===
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils import util_image  # 现在可以导入了

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# === 显式指定本地权重/参数文件 ===
MUSIQ_CKPT = os.environ.get("MUSIQ_CKPT", os.path.join(REPO_ROOT, "musiq_koniq_ckpt-e95806b9.pth"))
NIQE_PARAM = os.environ.get("NIQE_PARAM", os.path.join(REPO_ROOT, "niqe_modelparameters.mat"))
assert os.path.exists(MUSIQ_CKPT), f"MUSIQ ckpt not found: {MUSIQ_CKPT}"
assert os.path.exists(NIQE_PARAM), f"NIQE params not found: {NIQE_PARAM}"

# -------------------------
# 工具函数
# -------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

def list_image_files(folder: Path):
    folder = Path(folder)
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(files)

def get_group_suffix(stem: str, groups):
    """返回匹配的 group 后缀（如 '-3.0'），否则 None"""
    for g in groups:
        if stem.endswith(g):
            return g
    return None

def stem_base(stem: str, g: str):
    """stem 去掉末尾组后缀后的 base"""
    return stem[:-len(g)] if stem.endswith(g) else stem

def _to_uint8_img(t: torch.Tensor) -> Image.Image:
    """t: [1,C,H,W] in [0,1] or [0,255], return PIL RGB"""
    if t.max() <= 1.0 + 1e-6:
        t = (t * 255.0).clamp(0, 255)
    arr = t[0].detach().cpu().numpy().astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))  # HWC
    return Image.fromarray(arr)

def _hstack_with_text(img_left: Image.Image, img_right: Image.Image, text: str) -> Image.Image:
    """左右拼图并在顶部写信息"""
    h = max(img_left.height, img_right.height)
    w = img_left.width + img_right.width
    pad = 40  # 顶部留白写字
    canvas = Image.new("RGB", (w, h + pad), (0, 0, 0))
    canvas.paste(img_left, (0, pad))
    canvas.paste(img_right, (img_left.width, pad))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()
    draw.text((10, 10), text, fill=(255, 255, 255), font=font)
    return canvas

def _align_ref_tensor_resize_or_skip(ref_t: torch.Tensor, tgt_hw, mode: str) -> torch.Tensor:
    """
    将参考图 ref_t [1,C,H,W] 对齐到目标尺寸 tgt_hw=(H,W)
    mode: 'resize' | 'skip'
    - resize: 直接 resize 到目标尺寸
    - skip:   不做变换（由外层决定是否跳过该对）
    """
    _, _, H, W = ref_t.shape
    th, tw = tgt_hw
    if (H, W) == (th, tw):
        return ref_t
    if mode == "resize":
        return F.interpolate(ref_t, size=(th, tw), mode="bicubic", align_corners=False)
    return ref_t

def _make_temp_dir_with_symlinks(files, prefix="fid_subset_"):
    """
    为 FID 创建一个临时目录，只包含子集图片（优先 symlink，失败就 copy）
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix=prefix))
    for p in files:
        dst = tmp_dir / p.name
        try:
            os.symlink(str(p.resolve()), str(dst))
        except Exception:
            shutil.copy2(str(p), str(dst))
    return tmp_dir

# -------------------------
# 单组评测
# -------------------------
def evaluate_one_group(
    in_path: Path,
    ref_path: Path,
    pairs,                    # list[(pred_path, ref_path or None, base_key)]
    fr_align: str = "resize",
    save_vis_dir: str = None,
    compute_fid: bool = True,
):
    assert fr_align in {"resize", "skip"}, f"Invalid --fr_align: {fr_align}"

    if save_vis_dir is not None:
        os.makedirs(save_vis_dir, exist_ok=True)

    # NR-IQA 指标（对预测图计算）
    metric_nr = {
        "clipiqa": pyiqa.create_metric("clipiqa").to(device),
        "musiq": pyiqa.create_metric("musiq", pretrained_model_path=MUSIQ_CKPT).to(device),
        "niqe":  pyiqa.create_metric("niqe",  pretrained_model_path=NIQE_PARAM).to(device),
        "maniqa": pyiqa.create_metric("maniqa").to(device),
    }

    metric_fr = {}
    has_ref = (ref_path is not None)
    if has_ref:
        metric_fr = {
            "psnr":  pyiqa.create_metric("psnr", test_y_channel=True, color_space="ycbcr").to(device),
            "lpips": pyiqa.create_metric("lpips").to(device),
            "dists": pyiqa.create_metric("dists").to(device),
            "ssim":  pyiqa.create_metric("ssim", test_y_channel=True, color_space="ycbcr").to(device),
        }

    use_autocast = (device.type == "cuda")

    sum_nr = {k: 0.0 for k in metric_nr.keys()}
    sum_fr = {k: 0.0 for k in metric_fr.keys()}
    cnt_nr = 0
    cnt_fr = 0
    skipped_fr = 0

    for (pred_p, ref_p, base_key) in tqdm.tqdm(pairs, desc="Eval"):
        # --- load pred ---
        im_in = util_image.imread(pred_p, chn="rgb", dtype="float32")
        im_in_tensor = util_image.img2tensor(im_in).to(device)  # 1xCxHxW
        th, tw = im_in_tensor.shape[-2:]

        # --- NR ---
        for key, metric in metric_nr.items():
            if use_autocast:
                with torch.cuda.amp.autocast():
                    sum_nr[key] += float(metric(im_in_tensor).item())
            else:
                sum_nr[key] += float(metric(im_in_tensor).item())
        cnt_nr += 1

        # --- FR ---
        if has_ref and ref_p is not None:
            im_ref = util_image.imread(ref_p, chn="rgb", dtype="float32")
            im_ref_tensor = util_image.img2tensor(im_ref).to(device)

            if im_ref_tensor.shape[-2:] != (th, tw):
                if fr_align == "skip":
                    skipped_fr += 1
                    if save_vis_dir is not None:
                        left = _to_uint8_img(im_in_tensor.clamp(0, 1))
                        right = _to_uint8_img(im_ref_tensor.clamp(0, 1))
                        panel = _hstack_with_text(
                            left, right,
                            f"{base_key} | NOT ALIGNED (pred {th}x{tw} vs ref {im_ref_tensor.shape[-2]}x{im_ref_tensor.shape[-1]})"
                        )
                        panel.save(os.path.join(save_vis_dir, f"{pred_p.stem}_unaligned.png"))
                    continue
                else:
                    im_ref_tensor = _align_ref_tensor_resize_or_skip(im_ref_tensor, (th, tw), fr_align)

            per_image_scores = {}
            for key, metric in metric_fr.items():
                if use_autocast:
                    with torch.cuda.amp.autocast():
                        per_image_scores[key] = float(metric(im_in_tensor, im_ref_tensor).item())
                else:
                    per_image_scores[key] = float(metric(im_in_tensor, im_ref_tensor).item())
                sum_fr[key] += per_image_scores[key]
            cnt_fr += 1

            if save_vis_dir is not None:
                left = _to_uint8_img(im_in_tensor.clamp(0, 1))
                right = _to_uint8_img(im_ref_tensor.clamp(0, 1))
                txt = (f"{base_key} | PRED {im_in_tensor.shape[-2]}x{im_in_tensor.shape[-1]} "
                       f"| REF {im_ref_tensor.shape[-2]}x{im_ref_tensor.shape[-1]} "
                       f"| PSNR {per_image_scores['psnr']:.2f}  SSIM {per_image_scores['ssim']:.4f}  "
                       f"LPIPS {per_image_scores['lpips']:.3f}  DISTS {per_image_scores['dists']:.3f}")
                panel = _hstack_with_text(left, right, txt)
                panel.save(os.path.join(save_vis_dir, f"{pred_p.stem}.png"))
        else:
            if save_vis_dir is not None:
                left = _to_uint8_img(im_in_tensor.clamp(0, 1))
                panel = _hstack_with_text(left, left.copy(), f"{base_key} | NR-IQA only")
                panel.save(os.path.join(save_vis_dir, f"{pred_p.stem}.png"))

    # --- FID（按组子集算） ---
    fid_value = None
    tmp_pred_dir = None
    tmp_ref_dir = None
    if has_ref and compute_fid and len(pairs) > 0:
        # 仅使用参与配对的文件做 FID（更符合“按组评测”的预期）
        pred_files = [p for (p, r, _) in pairs if p is not None]
        ref_files  = [r for (p, r, _) in pairs if r is not None]
        if len(pred_files) > 0 and len(ref_files) > 0:
            try:
                tmp_pred_dir = _make_temp_dir_with_symlinks(pred_files, prefix="fid_pred_")
                tmp_ref_dir  = _make_temp_dir_with_symlinks(ref_files,  prefix="fid_ref_")
                fid_metric = pyiqa.create_metric("fid")
                fid_value = float(fid_metric(tmp_pred_dir, tmp_ref_dir))
            finally:
                # 清理临时目录
                if tmp_pred_dir is not None and tmp_pred_dir.exists():
                    shutil.rmtree(str(tmp_pred_dir), ignore_errors=True)
                if tmp_ref_dir is not None and tmp_ref_dir.exists():
                    shutil.rmtree(str(tmp_ref_dir), ignore_errors=True)

    # --- 汇总结果 ---
    results = {
        "count_pred": cnt_nr,
        "count_fr": cnt_fr,
        "skipped_fr": skipped_fr,
        "nr": {},
        "fr": {},
    }

    for k in sum_nr.keys():
        results["nr"][k] = (sum_nr[k] / max(cnt_nr, 1))

    for k in sum_fr.keys():
        results["fr"][k] = (sum_fr[k] / max(cnt_fr, 1)) if cnt_fr > 0 else None

    if fid_value is not None:
        results["fr"]["fid"] = fid_value
    else:
        results["fr"]["fid"] = None if has_ref else None

    # 写组内 metrics.txt
    if save_vis_dir is not None:
        lines = []
        lines.append(f"[info] count_pred: {cnt_nr}")
        lines.append(f"[info] count_fr: {cnt_fr}")
        lines.append(f"[info] skipped_fr: {skipped_fr}")
        lines.append("---- NR ----")
        for k, v in results["nr"].items():
            lines.append(f"{k}: {v:.5f}")
        if has_ref:
            lines.append("---- FR ----")
            for k, v in results["fr"].items():
                if v is None:
                    lines.append(f"{k}: N/A")
                elif k == "fid":
                    lines.append(f"{k}: {v:.2f}")
                else:
                    lines.append(f"{k}: {v:.5f}")
        metrics_path = os.path.join(save_vis_dir, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write("\n".join(lines))

    return results

# -------------------------
# 多组评测入口
# -------------------------
def evaluate_multi(
    in_path,
    ref_path,
    ntest,
    groups=("-3.0", "-4.0", "-5.0"),
    fr_align="resize",
    save_vis_dir=None,
    record_file=None,
    no_fid=False,
):
    in_path = Path(in_path)
    assert in_path.is_dir(), f"in_path is not a directory: {in_path}"

    ref_dir = None
    if ref_path is not None:
        ref_dir = Path(ref_path)
        assert ref_dir.is_dir(), f"ref_path is not a directory: {ref_dir}"

    pred_files = list_image_files(in_path)
    ref_files = list_image_files(ref_dir) if ref_dir is not None else []

    # 建立：group -> {base_key: path}
    pred_map = {g: {} for g in groups}
    ref_map  = {g: {} for g in groups}

    for p in pred_files:
        g = get_group_suffix(p.stem, groups)
        if g is None:
            continue
        b = stem_base(p.stem, g)
        pred_map[g][b] = p

    for p in ref_files:
        g = get_group_suffix(p.stem, groups)
        if g is None:
            continue
        b = stem_base(p.stem, g)
        ref_map[g][b] = p

    # 记录输出文件：默认放到 save_vis_dir/metrics_all.txt
    if record_file is None and save_vis_dir is not None:
        record_file = str(Path(save_vis_dir) / "metrics_all.txt")

    all_lines = []
    all_lines.append(f"=== Eval Multi Groups @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    all_lines.append(f"in_path: {in_path}")
    all_lines.append(f"ref_path: {ref_dir if ref_dir is not None else 'None'}")
    all_lines.append(f"groups: {list(groups)}")
    all_lines.append(f"fr_align: {fr_align}")
    all_lines.append(f"no_fid: {no_fid}")
    all_lines.append("")

    summary = {}

    for g in groups:
        # 取交集 base keys 来配对
        pred_keys = set(pred_map[g].keys())
        if ref_dir is not None:
            ref_keys = set(ref_map[g].keys())
            common = sorted(list(pred_keys & ref_keys))
        else:
            common = sorted(list(pred_keys))

        if ntest is not None:
            common = common[:ntest]

        pairs = []
        for b in common:
            pred_p = pred_map[g][b]
            ref_p = ref_map[g][b] if ref_dir is not None else None
            pairs.append((pred_p, ref_p, b))

        # 每组可视化目录
        group_vis_dir = None
        if save_vis_dir is not None:
            group_vis_dir = str(Path(save_vis_dir) / f"group{g}")
            os.makedirs(group_vis_dir, exist_ok=True)

        print(f"\n========== Group {g} ==========")
        print(f"[info] matched pairs: {len(pairs)} (pred-only: {len(pred_map[g])}, ref-only: {len(ref_map[g]) if ref_dir is not None else 0})")

        res = evaluate_one_group(
            in_path=in_path,
            ref_path=ref_dir,
            pairs=pairs,
            fr_align=fr_align,
            save_vis_dir=group_vis_dir,
            compute_fid=(not no_fid),
        )
        summary[g] = res

        # 输出到统一日志
        all_lines.append(f"========== Group {g} ==========")
        all_lines.append(f"[info] matched_pairs: {len(pairs)}")
        all_lines.append(f"[info] count_pred: {res['count_pred']}")
        all_lines.append(f"[info] count_fr: {res['count_fr']}")
        all_lines.append(f"[info] skipped_fr: {res['skipped_fr']}")
        all_lines.append("---- NR ----")
        for k, v in res["nr"].items():
            all_lines.append(f"{k}: {v:.5f}")
        if ref_dir is not None:
            all_lines.append("---- FR ----")
            for k, v in res["fr"].items():
                if v is None:
                    all_lines.append(f"{k}: N/A")
                elif k == "fid":
                    all_lines.append(f"{k}: {v:.2f}")
                else:
                    all_lines.append(f"{k}: {v:.5f}")
        all_lines.append("")

    # 写统一记录文件
    if record_file is not None:
        record_path = Path(record_file)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        with open(record_path, "a") as f:
            f.write("\n".join(all_lines) + "\n")
        print(f"\n[info] All-group metrics appended to: {record_path}")

    return summary

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_path", type=str, required=True)
    parser.add_argument("-r", "--ref_path", type=str, default=None)
    parser.add_argument("--ntest", type=int, default=None)

    parser.add_argument("--fr_align", type=str, default="resize",
                        choices=["resize", "skip"],
                        help="How to align reference image to input for FR metrics. "
                             "'resize' resizes REF to the size of PRED; 'skip' skips mismatched pairs.")

    parser.add_argument("--save_vis_dir", type=str, default=None,
                        help="If set, save side-by-side panels (PRED | REF-aligned) here. "
                             "Will create subfolders group-3.0/group-4.0/group-5.0.")

    parser.add_argument("--groups", type=str, default="-3.0,-4.0,-5.0",
                        help="Comma-separated group suffixes to evaluate, e.g. '-3.0,-4.0,-5.0'")

    parser.add_argument("--record_file", type=str, default=None,
                        help="If set, append all groups' metrics into this file. "
                             "If not set and --save_vis_dir is set, default to save_vis_dir/metrics_all.txt")

    parser.add_argument("--no_fid", action="store_true",
                        help="Disable FID computation (FID needs folder input; we create a temp subset folder).")

    args = parser.parse_args()
    groups = tuple([x.strip() for x in args.groups.split(",") if x.strip()])

    evaluate_multi(
        args.in_path,
        args.ref_path,
        args.ntest,
        groups=groups,
        fr_align=args.fr_align,
        save_vis_dir=args.save_vis_dir,
        record_file=args.record_file,
        no_fid=args.no_fid,
    )
