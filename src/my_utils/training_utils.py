import argparse
import json
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from glob import glob

import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from pathlib import Path
from torch.utils import data as data

from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor

from .data_process.realesrgan import RealESRGAN_degradation


def parse_args_paired_training(input_args=None):
    """Parse command-line arguments for paired training and inference."""
    parser = argparse.ArgumentParser()

    # === losses ===
    parser.add_argument("--gan_disc_type", default="vagan")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_lpips", default=5.0, type=float)
    parser.add_argument("--lambda_l2", default=2.0, type=float)

    parser.add_argument("--base_config", default="./configs/sr.yaml", type=str)
    parser.add_argument("--input_meta", default=None, type=str,
                        help="Path to the paired input/GT meta file used for inference.")

    # === eval ===
    parser.add_argument("--eval_freq", default=500, type=int)
    parser.add_argument("--save_val", action="store_true", default=False)
    parser.add_argument("--num_samples_eval", type=int, default=100)
    parser.add_argument("--viz_freq", type=int, default=100)

    # === model ===
    parser.add_argument("--sd_path")
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--de_net_path")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=32, type=int)
    parser.add_argument("--lora_rank_vae", default=32, type=int)
    parser.add_argument("--ablation_mode", type=str, default="full",
                        choices=["full", "no_eeg_all", "no_eeg_lora", "no_eeg_adapter", "no_lora", "vanilla_lora"],
                        help="Ablation switch inside CVSF.")
    parser.add_argument("--eeg_replace_mode", type=str, default="none",
                        choices=["none", "randn", "mismatch_visual", "other_eeg", "band_noise"],
                        help="Replace EEG embedding with equal-size control signals.")
    parser.add_argument("--eeg_noise_scale", type=float, default=1.0,
                        help="Noise scale for eeg_replace_mode=randn or band_noise.")
    # parser.add_argument("--eeg_band_start_ratio", type=float, default=0.2,
    #                     help="Start ratio of frequency band for band_noise in rFFT domain.")
    # parser.add_argument("--eeg_band_end_ratio", type=float, default=0.5,
    #                     help="End ratio of frequency band for band_noise in rFFT domain.")
    parser.add_argument("--eeg_random_seed", type=int, default=1234,
                        help="Random seed for reproducible EEG replacement.")
    parser.add_argument("--eeg_replace_target", type=str, default="lora",
                        choices=["lora", "prior", "both"],
                        help="Apply EEG replacement to LoRA branch, Prior branch, or both.")
    parser.add_argument("--prior_noise_scale", type=float, default=1.0,
                        help="Perturbation strength when replacement is applied to prior branch.")
    parser.add_argument("--other_eeg_pt_path", type=str, default=None,
                        help="Optional path to an external [N,1024] EEG embedding table for other_eeg replacement.")
    parser.add_argument("--eeg_retrieval_topk", type=int, default=8,
                        help="Top-k retrieval size for EEG prior injection.")
    parser.add_argument("--eeg_retrieval_temp", type=float, default=0.07,
                        help="Softmax temperature for EEG top-k retrieval weights.")
    parser.add_argument("--train_query_fuse_only", action="store_true",
                        help="Short finetune mode: optimize only EEG query/fuse related layers.")
    parser.add_argument("--strong_eeg_train", action="store_true",
                        help="Enable strong-EEG dependency training with margin regularization.")
    parser.add_argument("--strong_eeg_lambda", type=float, default=0.5,
                        help="Weight for strong-EEG margin regularization.")
    parser.add_argument("--strong_eeg_margin", type=float, default=0.005,
                        help="Target gap: mse(corrupted_eeg) - mse(normal_eeg) >= margin.")
    parser.add_argument("--strong_eeg_drop_mode", type=str, default="randn",
                        choices=["randn", "mismatch_visual", "other_eeg", "band_noise"],
                        help="EEG corruption mode for auxiliary strong-EEG branch.")
    parser.add_argument("--strong_eeg_drop_target", type=str, default="both",
                        choices=["lora", "prior", "both"],
                        help="Corruption target for auxiliary strong-EEG branch.")
    parser.add_argument("--neg_prob", default=0.05, type=float)
    parser.add_argument("--pos_prompt", type=str, default="A high-resolution, 8K, ultra-realistic image with natural lighting, sharp focus and vibrant colors.")
    parser.add_argument("--neg_prompt", type=str, default="inappropriate exposure, oil painting, cartoon, blur, dirty, messy, low quality, deformation, low resolution, oversmooth")

    # === training ===
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_training_epochs", type=int, default=50)
    parser.add_argument("--max_train_steps", type=int, default=50000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help='["linear","cosine","cosine_with_restarts","polynomial","constant","piecewise_constant","constant_with_warmup"]')
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=0.1)

    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true")


    parser.add_argument("--stage1_img_encoder_ckpt", type=str, required=True)
    parser.add_argument("--stage1_backbone", type=str, default="resnet34")
    parser.add_argument("--stage1_embed_dim", type=int, default=1024)
    parser.add_argument("--stage1_input_size", type=int, default=224)

    parser.add_argument("--prior_map_size", type=int, default=8)
    parser.add_argument("--prior_channels", type=int, default=256)

    # parser.add_argument("--lora_zero_parts", action="store_true")

    # === ablation ===
    parser.add_argument("--infer_batch_size", type=int, default=1)
    parser.add_argument("--shuffle_lora_in_batch", action="store_true")
    parser.add_argument("--shuffle_adapter_in_batch", action="store_true")
    parser.add_argument("--shuffle_seed", type=int, default=1234)
    parser.add_argument("--force_zero_prior", action="store_true")
    parser.add_argument("--force_zero_c", action="store_true")

    parser.add_argument("--force_noise_c", action="store_true")
    parser.add_argument("--noise_c_scale", type=float, default=1.0)
    parser.add_argument("--noise_c_seed", type=int, default=None)
    parser.add_argument("--noise_c_match_stats", action="store_true")

    parser.add_argument("--force_text_c", action="store_true")
    # parser.add_argument("--text_c_pool", type=str, default="mean", choices=["mean", "cls"])
    parser.add_argument("--text_c_detach", action="store_true")


    parser.add_argument("--force_band_noise_c", action="store_true")
    parser.add_argument("--band_noise_c_scale", type=float, default=1.0)
    parser.add_argument("--band_noise_c_seed", type=int, default=None)
    parser.add_argument("--band_start_ratio", type=float, default=0.2)
    parser.add_argument("--band_end_ratio", type=float, default=0.5)
    parser.add_argument("--band_noise_match_stats", action="store_true")



    parser.add_argument("--save_attention_vis", action="store_true")
    parser.add_argument("--attention_save_limit", type=int, default=4)
    parser.add_argument("--attention_save_stage", type=str, default="mid")
    parser.add_argument("--attention_save_dirname", type=str, default="attn_vis")

    parser.add_argument("--attention_save_keys", type=str, default="")
    parser.add_argument("--attn_overlay_alpha", type=float, default=0.75)
    parser.add_argument("--attn_cmap", type=str, default="turbo")

    parser.add_argument("--attn_robust_low", type=float, default=70)
    parser.add_argument("--attn_robust_high", type=float, default=99.5)

    parser.add_argument("--attn_heat_threshold", type=float, default=0.08)
    parser.add_argument("--attn_gamma", type=float, default=0.65)

    parser.add_argument("--attn_smooth_kernel", type=int, default=7)
    parser.add_argument("--attn_smooth_sigma", type=float, default=1.4)

    parser.add_argument("--save_blocky_attention", action="store_true")
    parser.add_argument("--save_overlay_input", action="store_true")
    parser.add_argument("--save_overlay_gt", action="store_true")


    parser.add_argument(
        "--c_guidance_type",
        type=str,
        default="image",
        choices=["image", "text", "depth", "edge", "seg"],
    )

    parser.add_argument(
        "--text_c_pool",
        type=str,
        default="mean",
        choices=["mean", "cls"],
    )

    parser.add_argument(
        "--seg_in_channels",
        type=int,
        default=1,
    )


    # === resume ===
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to accelerator state dir, e.g. runs/exp1/checkpoints/state_1000")

    # === inference specific ===
    parser.add_argument("--infer_full_image", action="store_true",
                        help="Use full-resolution inference for validation/test (no crop).")
    parser.add_argument("--tile_size", type=int, default=512,
                        help="Tile size for tiled inference. 0 -> disable tiling (one full forward).")
    parser.add_argument("--tile_overlap", type=int, default=64,
                        help="Overlap pixels between tiles to reduce seams.")

    # === degradation robustness ablations (inference) ===
    parser.add_argument("--deg_repeat_times", type=int, default=1,
                        help="Repeat degradation estimation N times per sample/tile.")
    parser.add_argument("--deg_fuse_mode", type=str, default="mean",
                        choices=["first", "mean", "median"],
                        help="How to fuse repeated degradation estimates.")
    parser.add_argument("--deg_input_noise_std", type=float, default=0.0,
                        help="Gaussian noise std added to DE-Net input (in [-1,1] image scale).")
    parser.add_argument("--deg_score_noise_std", type=float, default=0.0,
                        help="Gaussian noise std added to estimated degradation scores.")
    parser.add_argument("--deg_override_mode", type=str, default="none",
                        choices=["none", "zero", "random"],
                        help="Ablate degradation condition before feeding into CVSF.")
    parser.add_argument("--save_deg_scores", action="store_true",
                        help="Save per-image degradation scores to CSV during inference.")
    parser.add_argument("--deg_score_csv_name", type=str, default="deg_scores.csv",
                        help="Filename for per-image degradation score CSV under output_dir.")
    parser.add_argument("--lora_zero_parts", action="store_true",
                        help="Zero out all three LoRA modulation parts before writing back de_mod_0/1/2.")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


class PairedDataset(data.Dataset):
    """Paired dataset with RealESRGAN-style degradation for training."""
    def __init__(self, opt):
        super(PairedDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.crop_size = opt.get('crop_size', 512)
        if 'image_type' not in opt:
            opt['image_type'] = 'png'

        if 'meta_info' in opt:
            with open(self.opt['meta_info']) as fin:
                paths = [line.strip().split(' ')[0] for line in fin]
                self.paths = [v for v in paths]
            if 'meta_num' in opt:
                self.paths = sorted(self.paths)[:opt['meta_num']]
        if 'gt_path' in opt:
            if isinstance(opt['gt_path'], str):
                self.paths.extend(sorted([str(x) for x in Path(opt['gt_path']).rglob('*.' + opt['image_type'])]))
            else:
                for path in opt['gt_path']:
                    self.paths.extend(sorted([str(x) for x in Path(path).rglob('*.' + opt['image_type'])]))

        if 'imagenet_path' in opt:
            class_list = os.listdir(opt['imagenet_path'])
            for class_file in class_list:
                self.paths.extend(sorted([str(x) for x in Path(os.path.join(opt['imagenet_path'], class_file)).glob('*.JPEG')]))
        if 'face_gt_path' in opt:
            if isinstance(opt['face_gt_path'], str):
                face_list = sorted([str(x) for x in Path(opt['face_gt_path']).glob('*.'+opt['image_type'])])
                self.paths.extend(face_list[:opt['num_face']])
            else:
                face_list = sorted([str(x) for x in Path(opt['face_gt_path'][0]).glob('*.'+opt['image_type'])])
                self.paths.extend(face_list[:opt['num_face']])
                if len(opt['face_gt_path']) > 1:
                    for i in range(len(opt['face_gt_path'])-1):
                        self.paths.extend(sorted([str(x) for x in Path(opt['face_gt_path'][0]).glob('*.'+opt['image_type'])])[:opt['num_face']])

        if 'num_pic' in opt:
            random.shuffle(self.paths)
            self.paths = self.paths[:opt['num_pic']]
        if 'mul_num' in opt:
            self.paths = self.paths * opt['mul_num']

        deg_file_path = "params_realesrgan.yml"
        self.degradation = RealESRGAN_degradation(deg_file_path, device='cpu')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
            except (IOError, OSError):
                index = random.randint(0, self.__len__()-1)
                gt_path = self.paths[index]
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1

        img_gt = imfrombytes(img_bytes, float32=True)
        img_size_kb = os.path.getsize(gt_path) / 1024.0
        while img_gt.shape[0]*img_gt.shape[1] < 384*384 or img_size_kb < 100:
            index = random.randint(0, self.__len__()-1)
            gt_path = self.paths[index]
            time.sleep(0.1)
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_size_kb = os.path.getsize(gt_path) / 1024.0

        img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)

        h, w = img_gt.shape[0:2]
        cs = self.crop_size
        if h < cs or w < cs:
            pad_h = max(0, cs - h)
            pad_w = max(0, cs - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        if img_gt.shape[0] > cs or img_gt.shape[1] > cs:
            h, w = img_gt.shape[0:2]
            top = random.randint(0, h - cs)
            left = random.randint(0, w - cs)
            img_gt = img_gt[top:top + cs, left:left + cs, ...]

        output_t, img_t = self.degradation.degrade_process(img_gt, resize_bak=True)
        output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)
        img_t = TF.normalize(img_t, mean=[0.5], std=[0.5])
        output_t = TF.normalize(output_t, mean=[0.5], std=[0.5])

        return {'gt': output_t, 'lq': img_t, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)


def collect_image_paths(paths_or_str):
    """Collect image paths from a path or a list of paths."""
    image_types = ('jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG', 'ppm', 'PPM', 'bmp', 'BMP', 'tif')
    if isinstance(paths_or_str, (str, Path)):
        p = Path(paths_or_str)
        return sorted([f for f in p.rglob('*') if f.is_file() and f.suffix[1:].lower() in {e.lower() for e in image_types}])
    else:
        result = []
        for p in paths_or_str:
            P = Path(p)
            result.extend([f for f in P.rglob('*') if f.is_file() and f.suffix[1:].lower() in {e.lower() for e in image_types}])
        return sorted(result)


# class ExposurePairedDataset(data.Dataset):
#     """
#     def __init__(self, opt):
#         super(ExposurePairedDataset, self).__init__()
#         self.opt = opt
#         self.file_client = None
#         self.io_backend_opt = opt['io_backend']
#         self.crop_size = opt.get('crop_size', 512)
#         if 'image_type' not in opt:
#             opt['image_type'] = 'png'
#         self.phase = opt['phase']
#         self.pairs = []  # (gt_path, lq_path)

#         if 'meta_info' in opt:
#             with open(opt['meta_info'], 'r') as f:
#                 for line in f:
#                     parts = line.strip().split()
#                     if len(parts) < 2:
#                         continue
#                     gt_path, lq_path = parts[0], parts[1]
#                     self.pairs.append((gt_path, lq_path))
#             if 'meta_num' in opt:
#                 self.pairs = self.pairs[:opt['meta_num']]
#         else:
#             gt_list = collect_image_paths(opt['gt_path'])
#             lq_list = collect_image_paths(opt['lq_path'])
#             gt_dict = {p.name: str(p) for p in gt_list}
#             lq_dict = {p.name: str(p) for p in lq_list}
#             common = sorted(set(gt_dict.keys()).intersection(lq_dict.keys()))
#             if not common:
#             for name in common:
#                 self.pairs.append((gt_dict[name], lq_dict[name]))

#         if 'num_pic' in opt:
#             self.pairs = random.sample(self.pairs, opt['num_pic'])
#         if 'mul_num' in opt:
#             self.pairs = self.pairs * opt['mul_num']

#         deg_file_path = "params_realesrgan.yml"
#         self.degradation = RealESRGAN_degradation(deg_file_path, device='cpu')

#     def __getitem__(self, index):
#         if self.file_client is None:
#             self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

#         max_attempts = len(self.pairs)
#         attempts = 0
#         img_gt, img_lq = None, None
#         while attempts < max_attempts:
#             gt_path, lq_path = self.pairs[index]
#             try:
#                 img_gt = cv2.imread(gt_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
#                 img_lq = cv2.imread(lq_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
#                 if img_gt.shape != img_lq.shape:
#                     h, w = img_gt.shape[:2]
#                     img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
#                 break
#             except Exception as e:
#                 index = (index + 1) % len(self.pairs)
#                 attempts += 1
#         else:

#         infer_full = bool(self.opt.get('infer_full_image', True)) and \
#                      (self.phase.lower() in ['val', 'validation', 'test', 'infer'])
#         if infer_full:
#             if img_gt.shape != img_lq.shape:
#                 height, width = img_gt.shape[:2]
#                 img_lq = cv2.resize(img_lq, (width, height), interpolation=cv2.INTER_LINEAR)

#             img_lq = img2tensor([img_lq], bgr2rgb=True, float32=True)[0]
#             img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
#             img_lq = TF.normalize(img_lq, mean=[0.5], std=[0.5])
#             img_gt = TF.normalize(img_gt,  mean=[0.5], std=[0.5])
#             return {'gt': img_gt, 'gt_path': gt_path, 'lq': img_lq, 'lq_path': lq_path}

#         img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

#         h, w = img_gt.shape[:2]
#         cs = self.crop_size
#         if h < cs or w < cs:
#             pad_h, pad_w = max(0, cs - h), max(0, cs - w)
#             img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
#             img_lq = cv2.copyMakeBorder(img_lq, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
#         if img_gt.shape[0] > cs or img_gt.shape[1] > cs:
#             top = random.randint(0, img_gt.shape[0] - cs)
#             left = random.randint(0, img_gt.shape[1] - cs)
#             img_gt = img_gt[top:top+cs, left:left+cs, ...]
#             img_lq = img_lq[top:top+cs, left:left+cs, ...]

#         if random.random() < 0.3:
#             _, img_lq_ = self.degradation.degrade_process(img_lq, resize_bak=True)
#             img_lq_ = img_lq_.squeeze(0)
#             img_lq = TF.normalize(img_lq_, mean=[0.5], std=[0.5])
#         else:
#             img_lq = img2tensor([img_lq], bgr2rgb=True, float32=True)[0]
#             img_lq = TF.normalize(img_lq, mean=[0.5], std=[0.5])

#         img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
#         img_gt = TF.normalize(img_gt, mean=[0.5], std=[0.5])

#         return {'gt': img_gt, 'gt_path': gt_path, 'lq': img_lq, 'lq_path': lq_path}

#     def __len__(self):
#         return len(self.pairs)


class ExposurePairedDataset(data.Dataset):
    """Load paired GT/LQ images and optional guidance maps."""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.crop_size = opt.get('crop_size', 512)

        if 'image_type' not in opt:
            opt['image_type'] = 'png'

        self.phase = str(opt['phase']).lower()
        self.pairs = []   # list[dict]: {"gt":..., "lq":..., "depth":..., "seg":..., "edge":...}

        self.depth_root = opt.get('depth_path', None)
        self.seg_root = opt.get('seg_path', None)
        self.edge_root = opt.get('edge_path', None)


        if 'meta_info' in opt:
            self._build_pairs_from_meta(opt)
        else:
            self._build_pairs_from_folders(opt)

        if 'num_pic' in opt:
            self.pairs = random.sample(self.pairs, opt['num_pic'])
        if 'mul_num' in opt:
            self.pairs = self.pairs * opt['mul_num']

        deg_file_path = "params_realesrgan.yml"
        self.degradation = RealESRGAN_degradation(deg_file_path, device='cpu')

    def _safe_collect_paths(self, root):
        if root is None:
            return []
        if not os.path.exists(root):
            return []

        try:
            paths = collect_image_paths(root)
            if len(paths) > 0:
                return [str(p) for p in paths]
        except Exception:
            pass

        exts = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"]
        out = []
        for ext in exts:
            out.extend(glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True))
            out.extend(glob.glob(os.path.join(root, "**", f"*.{ext.upper()}"), recursive=True))
        out = sorted(list(set(out)))
        return out

    def _make_name_dict(self, file_list):
        """
        杩斿洖:
          by_name:  {filename -> fullpath}
          by_stem:  {stem -> [fullpath1, fullpath2, ...]}
        """
        by_name = {}
        by_stem = {}
        for fp in file_list:
            name = os.path.basename(fp)
            stem = os.path.splitext(name)[0]
            by_name[name] = fp
            by_stem.setdefault(stem, []).append(fp)
        return by_name, by_stem

    def _choose_best_depth(self, base_name, depth_by_name, depth_files):
        """Find the best matching depth map for an image name."""
        stem, ext = os.path.splitext(base_name)

        cand = depth_by_name.get(base_name, None)
        if cand is not None:
            return cand

        cand = depth_by_name.get(f"{stem}_depth{ext}", None)
        if cand is not None:
            return cand

        prefix = f"{stem}-"
        cands = []
        for fp in depth_files:
            name = os.path.basename(fp)
            if name.startswith(prefix) and name.endswith(ext):
                cands.append(fp)

        if len(cands) == 1:
            return cands[0]
        if len(cands) > 1:
            return sorted(cands)[0]

        return None

    def _choose_best_seg(self, base_name, seg_by_name, seg_files):
        """Find the best matching segmentation map for an image name."""
        stem, ext = os.path.splitext(base_name)

        cand = seg_by_name.get(f"{stem}_seg{ext}", None)
        if cand is not None:
            return cand

        cand = seg_by_name.get(base_name, None)
        if cand is not None:
            return cand

        cands = []
        prefix = f"{stem}-"
        suffix = f"_seg{ext}"
        for fp in seg_files:
            name = os.path.basename(fp)
            if name.startswith(prefix) and name.endswith(suffix):
                cands.append(fp)

        if len(cands) == 1:
            return cands[0]
        if len(cands) > 1:
            return sorted(cands)[0]

        cands = []
        for fp in seg_files:
            name = os.path.basename(fp)
            if name.startswith(prefix) and name.endswith(ext):
                cands.append(fp)

        if len(cands) == 1:
            return cands[0]
        if len(cands) > 1:
            return sorted(cands)[0]

        return None

    def _choose_best_edge(self, base_name, edge_by_name, edge_files):
        """Find the best matching edge map for an image name."""
        stem, ext = os.path.splitext(base_name)

        cand = edge_by_name.get(base_name, None)
        if cand is not None:
            return cand

        cand = edge_by_name.get(f"{stem}_edge{ext}", None)
        if cand is not None:
            return cand

        prefix = f"{stem}-"
        cands = []
        for fp in edge_files:
            name = os.path.basename(fp)
            if name.startswith(prefix) and name.endswith(ext):
                cands.append(fp)

        if len(cands) == 1:
            return cands[0]
        if len(cands) > 1:
            return sorted(cands)[0]

        return None

    def _build_pairs_from_meta(self, opt):
        depth_files = self._safe_collect_paths(self.depth_root)
        seg_files = self._safe_collect_paths(self.seg_root)
        edge_files = self._safe_collect_paths(self.edge_root)

        depth_by_name, _ = self._make_name_dict(depth_files)
        seg_by_name, _ = self._make_name_dict(seg_files)
        edge_by_name, _ = self._make_name_dict(edge_files)

        with open(opt['meta_info'], 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                gt_path, lq_path = parts[0], parts[1]
                base_name = os.path.basename(gt_path)

                depth_path = self._choose_best_depth(base_name, depth_by_name, depth_files) if self.depth_root else None
                seg_path = self._choose_best_seg(base_name, seg_by_name, seg_files) if self.seg_root else None
                edge_path = self._choose_best_edge(base_name, edge_by_name, edge_files) if self.edge_root else None

                self.pairs.append({
                    "gt": gt_path,
                    "lq": lq_path,
                    "depth": depth_path,
                    "seg": seg_path,
                    "edge": edge_path,
                })

        if 'meta_num' in opt:
            self.pairs = self.pairs[:opt['meta_num']]

    def _build_pairs_from_folders(self, opt):
        gt_list = self._safe_collect_paths(opt['gt_path'])
        lq_list = self._safe_collect_paths(opt['lq_path'])
        depth_list = self._safe_collect_paths(self.depth_root) if self.depth_root else []
        seg_list = self._safe_collect_paths(self.seg_root) if self.seg_root else []
        edge_list = self._safe_collect_paths(self.edge_root) if self.edge_root else []

        gt_dict = {os.path.basename(p): p for p in gt_list}
        lq_dict = {os.path.basename(p): p for p in lq_list}
        depth_by_name, _ = self._make_name_dict(depth_list)
        seg_by_name, _ = self._make_name_dict(seg_list)
        edge_by_name, _ = self._make_name_dict(edge_list)

        common = sorted(set(gt_dict.keys()).intersection(lq_dict.keys()))
        if not common:
            raise ValueError("No paired GT/LQ images found with matching file names.")

        for name in common:
            depth_path = self._choose_best_depth(name, depth_by_name, depth_list) if self.depth_root else None
            seg_path = self._choose_best_seg(name, seg_by_name, seg_list) if self.seg_root else None
            edge_path = self._choose_best_edge(name, edge_by_name, edge_list) if self.edge_root else None

            self.pairs.append({
                "gt": gt_dict[name],
                "lq": lq_dict[name],
                "depth": depth_path,
                "seg": seg_path,
                "edge": edge_path,
            })

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        max_attempts = len(self.pairs)
        attempts = 0
        img_gt, img_lq = None, None
        depth, seg, edge = None, None, None

        while attempts < max_attempts:
            sample = self.pairs[index]
            gt_path = sample["gt"]
            lq_path = sample["lq"]
            depth_path = sample.get("depth", None)
            seg_path = sample.get("seg", None)
            edge_path = sample.get("edge", None)

            try:
                img_gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
                img_lq = cv2.imread(lq_path, cv2.IMREAD_COLOR)

                if img_gt is None or img_lq is None:
                    raise ValueError(f"璇诲彇澶辫触: gt={gt_path}, lq={lq_path}")

                img_gt = img_gt.astype(np.float32) / 255.
                img_lq = img_lq.astype(np.float32) / 255.

                if img_gt.shape != img_lq.shape:
                    h, w = img_gt.shape[:2]
                    img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

                depth = None
                if depth_path is not None and os.path.exists(depth_path):
                    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
                    if depth is not None:
                        if depth.shape[:2] != img_gt.shape[:2]:
                            h, w = img_gt.shape[:2]
                            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
                        depth = depth.astype(np.float32) / 255.

                seg = None
                if seg_path is not None and os.path.exists(seg_path):
                    seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
                    if seg is not None:
                        if seg.shape[:2] != img_gt.shape[:2]:
                            h, w = img_gt.shape[:2]
                            seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
                        seg = seg.astype(np.float32) / 255.

                edge = None
                if edge_path is not None and os.path.exists(edge_path):
                    edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
                    if edge is not None:
                        if edge.shape[:2] != img_gt.shape[:2]:
                            h, w = img_gt.shape[:2]
                            edge = cv2.resize(edge, (w, h), interpolation=cv2.INTER_LINEAR)
                        edge = edge.astype(np.float32) / 255.

                break

            except Exception as e:
                index = (index + 1) % len(self.pairs)
                attempts += 1
        else:
            raise ValueError("All samples failed to load. Please check the dataset paths.")

        infer_full = bool(self.opt.get('infer_full_image', True)) and \
                     (self.phase in ['val', 'validation', 'test', 'infer'])

        if infer_full:
            if img_gt.shape != img_lq.shape:
                h, w = img_gt.shape[:2]
                img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

            img_lq = img2tensor([img_lq], bgr2rgb=True, float32=True)[0]
            img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]

            img_lq = TF.normalize(img_lq, mean=[0.5], std=[0.5])
            img_gt = TF.normalize(img_gt, mean=[0.5], std=[0.5])

            out = {
                'gt': img_gt,
                'gt_path': gt_path,
                'lq': img_lq,
                'lq_path': lq_path,
            }

            if depth is not None:
                depth = torch.from_numpy(depth).unsqueeze(0).float()
                out['depth'] = depth

            if seg is not None:
                seg = torch.from_numpy(seg).unsqueeze(0).float()
                out['seg'] = seg

            if edge is not None:
                edge = torch.from_numpy(edge).unsqueeze(0).float()
                out['edge'] = edge

            return out

        img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        h, w = img_gt.shape[:2]
        cs = self.crop_size

        if h < cs or w < cs:
            pad_h, pad_w = max(0, cs - h), max(0, cs - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            img_lq = cv2.copyMakeBorder(img_lq, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

            if depth is not None:
                depth = cv2.copyMakeBorder(depth, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            if seg is not None:
                seg = cv2.copyMakeBorder(seg, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            if edge is not None:
                edge = cv2.copyMakeBorder(edge, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

        if img_gt.shape[0] > cs or img_gt.shape[1] > cs:
            top = random.randint(0, img_gt.shape[0] - cs)
            left = random.randint(0, img_gt.shape[1] - cs)

            img_gt = img_gt[top:top + cs, left:left + cs, ...]
            img_lq = img_lq[top:top + cs, left:left + cs, ...]

            if depth is not None:
                depth = depth[top:top + cs, left:left + cs]
            if seg is not None:
                seg = seg[top:top + cs, left:left + cs]
            if edge is not None:
                edge = edge[top:top + cs, left:left + cs]

        if random.random() < 0.3:
            _, img_lq_ = self.degradation.degrade_process(img_lq, resize_bak=True)
            img_lq_ = img_lq_.squeeze(0)
            img_lq = TF.normalize(img_lq_, mean=[0.5], std=[0.5])
        else:
            img_lq = img2tensor([img_lq], bgr2rgb=True, float32=True)[0]
            img_lq = TF.normalize(img_lq, mean=[0.5], std=[0.5])

        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        img_gt = TF.normalize(img_gt, mean=[0.5], std=[0.5])

        out = {
            'gt': img_gt,
            'gt_path': gt_path,
            'lq': img_lq,
            'lq_path': lq_path,
        }

        if depth is not None:
            depth = torch.from_numpy(depth).unsqueeze(0).float()
            out['depth'] = depth

        if seg is not None:
            seg = torch.from_numpy(seg).unsqueeze(0).float()
            out['seg'] = seg

        if edge is not None:
            edge = torch.from_numpy(edge).unsqueeze(0).float()
            out['edge'] = edge

        return out

    def __len__(self):
        return len(self.pairs)
