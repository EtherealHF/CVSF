import os
import numpy as np
import cv2
import glob
import math
import yaml
import random
from collections import OrderedDict
import torch
import torch.nn.functional as F

from basicsr.data.transforms import augment
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.utils import DiffJPEG, USMSharp, img2tensor, tensor2img
from basicsr.utils.img_process_util import filter2D
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize, rgb_to_grayscale)

from .exposure_process import *
from .exposure_unprocess import *

cur_path = os.path.dirname(os.path.abspath(__file__))

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def opt_parse(opt_path):
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)  # ignore_security_alert_wait_for_fix RCE

    return opt

class RealESRGAN_degradation(object):
    def __init__(self, opt_name='params_realesrgan.yml', device='cpu'):
        opt_path = f'{cur_path}/{opt_name}'
        self.opt = opt_parse(opt_path)
        self.device = device #torch.device('cpu')
        optk = self.opt['kernel_info']       

        # blur settings for the first degradation
        self.blur_kernel_size = optk['blur_kernel_size']
        self.kernel_list = optk['kernel_list']
        self.kernel_prob = optk['kernel_prob']
        self.blur_sigma = optk['blur_sigma']
        self.betag_range = optk['betag_range']
        self.betap_range = optk['betap_range']
        self.sinc_prob = optk['sinc_prob']

        # blur settings for the second degradation
        self.blur_kernel_size2 = optk['blur_kernel_size2']
        self.kernel_list2 = optk['kernel_list2']
        self.kernel_prob2 = optk['kernel_prob2']
        self.blur_sigma2 = optk['blur_sigma2']
        self.betag_range2 = optk['betag_range2']
        self.betap_range2 = optk['betap_range2']
        self.sinc_prob2 = optk['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = optk['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

        self.jpeger = DiffJPEG(differentiable=False).to(self.device)
        self.usm_shaper = USMSharp().to(self.device)
    
    def color_jitter_pt(self, img, brightness, contrast, saturation, hue):
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img

    def random_augment(self, img_gt):
        # random horizontal flip
        # img_gt, status = augment(img_gt, hflip=True, rotation=False, return_status=True)
        # img_gt, status = augment(img_gt, hflip=True, rotation=False, return_status=True)
        """
        # random color jitter 
        if np.random.uniform() < self.opt['color_jitter_prob']:
            jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
            img_gt = img_gt + jitter_val
            img_gt = np.clip(img_gt, 0, 1)    

        # random grayscale
        if np.random.uniform() < self.opt['gray_prob']:
            #img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_RGB2GRAY)
            img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])
        """
        # BGR to RGB, HWC to CHW, numpy to tensor
        # img_gt = img2tensor([img_gt], bgr2rgb=False, float32=True)[0].unsqueeze(0)
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0].unsqueeze(0)

        return img_gt

    def random_kernels(self):
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                    self.kernel_list,
                    self.kernel_prob,
                    kernel_size,
                    self.blur_sigma,
                    self.blur_sigma, [-math.pi, math.pi],
                    self.betag_range,
                    self.betap_range,
                    noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- sinc kernel ------------------------------------- #
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2) 

        return kernel, kernel2, sinc_kernel

    # @torch.no_grad()
    # def degrade_process(self, img_gt, resize_bak=False):
    #     img_gt = self.random_augment(img_gt)
    #     kernel1, kernel2, sinc_kernel = self.random_kernels()
    #     img_gt, kernel1, kernel2, sinc_kernel = img_gt.to(self.device), kernel1.to(self.device), kernel2.to(self.device), sinc_kernel.to(self.device)
    #     #img_gt = self.usm_shaper(img_gt) # shaper gt
    #     ori_h, ori_w = img_gt.size()[2:4]

    #     # #scale_final = random.randint(4, 16)
    #     # scale_final = 4

        

    #     # ----------------------- The first degradation process ----------------------- #
        
    #     # blur
    #     out = filter2D(img_gt, kernel1)



    #     # # random resize
    #     # updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
    #     # if updown_type == 'up':
    #     #     scale = np.random.uniform(1, self.opt['resize_range'][1])
    #     # elif updown_type == 'down':
    #     #     scale = np.random.uniform(self.opt['resize_range'][0], 1)
    #     # else:
    #     #     scale = 1
    #     # mode = random.choice(['area', 'bilinear', 'bicubic'])
    #     # out = F.interpolate(out, scale_factor=scale, mode=mode)
    #     # noise
    #     gray_noise_prob = self.opt['gray_noise_prob']
    #     if np.random.uniform() < self.opt['gaussian_noise_prob']:
    #         out = random_add_gaussian_noise_pt(
    #             out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
    #     else:
    #         out = random_add_poisson_noise_pt(
    #             out,
    #             scale_range=self.opt['poisson_scale_range'],
    #             gray_prob=gray_noise_prob,
    #             clip=True,
    #             rounds=False)
    #     # JPEG compression
    #     jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
    #     out = torch.clamp(out, 0, 1)
    #     out = self.jpeger(out, quality=jpeg_p)

    #     # ----------------------- The second degradation process ----------------------- #
    #     # blur
    #     if np.random.uniform() < self.opt['second_blur_prob']:
    #         out = filter2D(out, kernel2)
    #     # # random resize
    #     # updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
    #     # if updown_type == 'up':
    #     #     scale = np.random.uniform(1, self.opt['resize_range2'][1])
    #     # elif updown_type == 'down':
    #     #     scale = np.random.uniform(self.opt['resize_range2'][0], 1)
    #     # else:
    #     #     scale = 1
    #     # mode = random.choice(['area', 'bilinear', 'bicubic'])
    #     # out = F.interpolate(
    #     #     out, size=(int(ori_h / scale_final * scale), int(ori_w / scale_final * scale)), mode=mode)
    #     # noise
    #     gray_noise_prob = self.opt['gray_noise_prob2']
    #     if np.random.uniform() < self.opt['gaussian_noise_prob2']:
    #         out = random_add_gaussian_noise_pt(
    #             out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
    #     else:
    #         out = random_add_poisson_noise_pt(
    #             out,
    #             scale_range=self.opt['poisson_scale_range2'],
    #             gray_prob=gray_noise_prob,
    #             clip=True,
    #             rounds=False)

    #     # JPEG compression + the final sinc filter
    #     # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
    #     # as one operation.
    #     # We consider two orders:
    #     #   1. [resize back + sinc filter] + JPEG compression
    #     #   2. JPEG compression + [resize back + sinc filter]
    #     # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
    #     if np.random.uniform() < 0.5:
    #         # # resize back + the final sinc filter
    #         # mode = random.choice(['area', 'bilinear', 'bicubic'])
    #         # out = F.interpolate(out, size=(ori_h // scale_final, ori_w // scale_final), mode=mode)
    #         out = filter2D(out, sinc_kernel)
    #         # JPEG compression
    #         jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
    #         out = torch.clamp(out, 0, 1)
    #         out = self.jpeger(out, quality=jpeg_p)
    #     else:
    #         # JPEG compression
    #         jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
    #         out = torch.clamp(out, 0, 1)
    #         out = self.jpeger(out, quality=jpeg_p)
    #         # # resize back + the final sinc filter
    #         # mode = random.choice(['area', 'bilinear', 'bicubic'])
    #         # out = F.interpolate(out, size=(ori_h // scale_final, ori_w // scale_final), mode=mode)
    #         out = filter2D(out, sinc_kernel)

    #     # if np.random.uniform() < self.opt['gray_prob']:
    #     #     out = rgb_to_grayscale(out, num_output_channels=1)

    #     if np.random.uniform() < self.opt['color_jitter_prob']:
    #         brightness = self.opt.get('brightness', (0.5, 1.5))
    #         contrast = self.opt.get('contrast', (0.5, 1.5))
    #         saturation = self.opt.get('saturation', (0, 1.5))
    #         hue = self.opt.get('hue', (-0.1, 0.1))
    #         out = self.color_jitter_pt(out, brightness, contrast, saturation, hue)

    #     # adjust exposure
    #     out = self.exposure_adjust(out) ### 调整光照

    #     if resize_bak:
    #         mode = random.choice(['area', 'bilinear', 'bicubic'])
    #         out = F.interpolate(out, size=(ori_h, ori_w), mode=mode)
    #     # clamp and round
    #     img_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

    #     return img_gt, img_lq

    @torch.no_grad()
    def degrade_process(self, img_gt, resize_bak=False):
        img_gt = self.random_augment(img_gt)
        kernel1, kernel2, sinc_kernel = self.random_kernels()
        img_gt, kernel1, kernel2, sinc_kernel = img_gt.to(self.device), kernel1.to(self.device), kernel2.to(self.device), sinc_kernel.to(self.device)
        #img_gt = self.usm_shaper(img_gt) # shaper gt
        ori_h, ori_w = img_gt.size()[2:4]

        # #scale_final = random.randint(4, 16)
        # scale_final = 4

        

        # ----------------------- The first degradation process ----------------------- #
        
        # blur
        out = filter2D(img_gt, kernel1)



        # # random resize
        # updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
        # if updown_type == 'up':
        #     scale = np.random.uniform(1, self.opt['resize_range'][1])
        # elif updown_type == 'down':
        #     scale = np.random.uniform(self.opt['resize_range'][0], 1)
        # else:
        #     scale = 1
        # mode = random.choice(['area', 'bilinear', 'bicubic'])
        # out = F.interpolate(out, scale_factor=scale, mode=mode)
        # noise
        gray_noise_prob = self.opt['gray_noise_prob']
        if np.random.uniform() < self.opt['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.opt['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
        out = torch.clamp(out, 0, 1)
        out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.opt['second_blur_prob']:
            out = filter2D(out, kernel2)
        # # random resize
        # updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
        # if updown_type == 'up':
        #     scale = np.random.uniform(1, self.opt['resize_range2'][1])
        # elif updown_type == 'down':
        #     scale = np.random.uniform(self.opt['resize_range2'][0], 1)
        # else:
        #     scale = 1
        # mode = random.choice(['area', 'bilinear', 'bicubic'])
        # out = F.interpolate(
        #     out, size=(int(ori_h / scale_final * scale), int(ori_w / scale_final * scale)), mode=mode)
        # noise
        gray_noise_prob = self.opt['gray_noise_prob2']
        if np.random.uniform() < self.opt['gaussian_noise_prob2']:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.opt['poisson_scale_range2'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # # resize back + the final sinc filter
            # mode = random.choice(['area', 'bilinear', 'bicubic'])
            # out = F.interpolate(out, size=(ori_h // scale_final, ori_w // scale_final), mode=mode)
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # # resize back + the final sinc filter
            # mode = random.choice(['area', 'bilinear', 'bicubic'])
            # out = F.interpolate(out, size=(ori_h // scale_final, ori_w // scale_final), mode=mode)
            out = filter2D(out, sinc_kernel)

        # if np.random.uniform() < self.opt['gray_prob']:
        #     out = rgb_to_grayscale(out, num_output_channels=1)

        if np.random.uniform() < self.opt['color_jitter_prob']:
            brightness = self.opt.get('brightness', (0.5, 1.5))
            contrast = self.opt.get('contrast', (0.5, 1.5))
            saturation = self.opt.get('saturation', (0, 1.5))
            hue = self.opt.get('hue', (-0.1, 0.1))
            out = self.color_jitter_pt(out, brightness, contrast, saturation, hue)

        # adjust exposure
        # out = self.exposure_adjust(out) ### 调整光照


        # if resize_bak:
        #     mode = random.choice(['area', 'bilinear', 'bicubic'])
        #     out = F.interpolate(out, size=(ori_h, ori_w), mode=mode)
        # # clamp and round
        # img_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        return img_gt, out

    @torch.no_grad()
    def degrade_process_1(self, img_gt, resize_bak=False):
        img_gt = self.random_augment(img_gt)
        kernel1, kernel2, sinc_kernel = self.random_kernels()
        img_gt, kernel1, kernel2, sinc_kernel = img_gt.to(self.device), kernel1.to(self.device), kernel2.to(self.device), sinc_kernel.to(self.device)
        #img_gt = self.usm_shaper(img_gt) # shaper gt
        ori_h, ori_w = img_gt.size()[2:4]

        out=img_gt

        # # JPEG compression
        # jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
        # out = torch.clamp(out, 0, 1)
        # out = self.jpeger(out, quality=jpeg_p)

        # # ----------------------- The second degradation process ----------------------- #
        # # blur
        # if np.random.uniform() < self.opt['second_blur_prob']:
        #     out = filter2D(out, kernel2)

        # # noise
        # gray_noise_prob = self.opt['gray_noise_prob2']
        # if np.random.uniform() < self.opt['gaussian_noise_prob2']:
        #     out = random_add_gaussian_noise_pt(
        #         out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        # else:
        #     out = random_add_poisson_noise_pt(
        #         out,
        #         scale_range=self.opt['poisson_scale_range2'],
        #         gray_prob=gray_noise_prob,
        #         clip=True,
        #         rounds=False)

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # # resize back + the final sinc filter
            # mode = random.choice(['area', 'bilinear', 'bicubic'])
            # out = F.interpolate(out, size=(ori_h // scale_final, ori_w // scale_final), mode=mode)
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # # resize back + the final sinc filter
            # mode = random.choice(['area', 'bilinear', 'bicubic'])
            # out = F.interpolate(out, size=(ori_h // scale_final, ori_w // scale_final), mode=mode)
            out = filter2D(out, sinc_kernel)

        # if np.random.uniform() < self.opt['gray_prob']:
        #     out = rgb_to_grayscale(out, num_output_channels=1)

        # if np.random.uniform() < self.opt['color_jitter_prob']:
        #     brightness = self.opt.get('brightness', (0.5, 1.5))
        #     contrast = self.opt.get('contrast', (0.5, 1.5))
        #     saturation = self.opt.get('saturation', (0, 1.5))
        #     hue = self.opt.get('hue', (-0.1, 0.1))
        #     out = self.color_jitter_pt(out, brightness, contrast, saturation, hue)

        # adjust exposure
        out = self.exposure_adjust(out) ### 调整光照

        if resize_bak:
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h, ori_w), mode=mode)
        # clamp and round
        img_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        return img_gt, img_lq
    
 
    @torch.no_grad()
    def exposure_adjust(self, img):
        # 如果输入没有 batch 维度，则增加 batch 维度 (B, C, H, W)
        if img.dim() == 3:
            img = img.unsqueeze(0)
            
        # 进行 unprocess 操作，返回 raw Bayer 图像和元数据
        img_tensor, metadata = unprocess(img)
        
        # 处理元数据，使 red_gain、blue_gain 和 cam2rgb 均扩展为 batch 形式
        red_gain = metadata['red_gain']
        if red_gain.dim() == 1:
            red_gain = red_gain.unsqueeze(0)
        if red_gain.shape[-1] != 1:
            red_gain = red_gain.view(-1, 1)
        
        blue_gain = metadata['blue_gain']
        if blue_gain.dim() == 1:
            blue_gain = blue_gain.unsqueeze(0)
        if blue_gain.shape[-1] != 1:
            blue_gain = blue_gain.view(-1, 1)
        
        cam2rgb = metadata['cam2rgb']
        if cam2rgb.dim() == 2:
            cam2rgb = cam2rgb.unsqueeze(0)
        
        # 随机选择曝光因子：50% 的概率选择较小值（降低曝光），50% 的概率选择较大值（提高曝光）
        flag = np.random.rand()
        if flag < 0.5:
            numbers = np.linspace(0.01, 0.05, 10)
            selected = np.random.choice(numbers)
        else:
            numbers = np.linspace(5, 7, 10)
            selected = np.random.choice(numbers)
        


        # # --------------------
        # # 4. 生成带渐变的椭圆掩码
        # # --------------------
        # B, C, H, W = img_tensor.shape
        
        # # 随机选择椭圆中心 (cx, cy)，以及椭圆半径 (rx, ry)
        # # 这里用 0.3~0.7 作为中心相对位置，0.2~0.4 作为相对半径，可自行调整
        # cx = random.uniform(0.3, 0.7) * W
        # cy = random.uniform(0.3, 0.7) * H
        # rx = random.uniform(0.2, 0.4) * W
        # ry = random.uniform(0.2, 0.4) * H
        
        # # 创建坐标网格 (H, W)，计算每个像素相对椭圆中心的归一化距离 d
        # # d < 1 表示在椭圆内，d = 1 表示正好在椭圆边缘
        # y = torch.arange(H, device=img_tensor.device).view(-1, 1).float()  # shape (H, 1)
        # x = torch.arange(W, device=img_tensor.device).view(1, -1).float()  # shape (1, W)
        # d = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2  # shape (H, W)
        
        # # 我们希望在椭圆边缘附近有一个平滑过渡区
        # # 这里定义 soft_edge 作为过渡带的宽度
        # soft_edge = 0.2  # 可根据需求调整
        # # 在椭圆内 (d < 1) => mask = 1
        # # 在椭圆外较远 (d > 1 + soft_edge) => mask = 0
        # # 在 1 <= d <= 1 + soft_edge 之间做线性插值
        
        # mask = torch.zeros_like(d)  # shape (H, W)
        # inside = (d < 1.0)
        # outside = (d > 1.0 + soft_edge)
        # transition = ~inside & ~outside  # 即 1.0 <= d <= 1.0 + soft_edge
        
        # mask[inside] = 1.0
        # # 线性插值: 当 d=1 => mask=1, 当 d=1+soft_edge => mask=0
        # mask[transition] = 1.0 - (d[transition] - 1.0) / soft_edge
        # mask[outside] = 0.0
        
        # # 扩展到 (B, 1, H, W)，让同一个椭圆作用到所有 batch
        # mask = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
        
        # # --------------------
        # # 5. 设置过曝与欠曝倍数
        # # --------------------
        # # 这里根据需求随机一个合理范围：过曝 2~3 倍，欠曝 0.5~0.8 倍
        # over_factor = random.uniform(2.0, 3.0)
        # under_factor = random.uniform(0.5, 0.8)
        
        # # 将 mask 视为“过曝权重”，根据它插值
        # # 方案1: mask=1 代表过曝区，mask=0 代表欠曝区
        # # exposure_map = mask * over_factor + (1 - mask) * under_factor
        
        # # 如果想让过曝区和欠曝区都带有渐变，可以再对 mask 做非线性映射
        # # 这里用最简单的线性组合
        # exposure_map = mask * over_factor + (1 - mask) * under_factor
        
        # # 6. 应用曝光调整到 raw Bayer 图像
        # img_tensor_adjusted = img_tensor * exposure_map



        # 调整曝光：直接乘以随机选取的因子
        img_tensor_adjusted = img_tensor * selected
        
        # 利用 forward processing 恢复为 sRGB 图像
        rgb_image_in = process(img_tensor_adjusted, red_gain, blue_gain, cam2rgb)
        
        return rgb_image_in
