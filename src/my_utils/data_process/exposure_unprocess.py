# # coding=utf-8
# # Copyright 2019 The Google Research Authors.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# """Unprocesses sRGB images into realistic raw data.

# Unprocessing Images for Learned Raw Denoising
# http://timothybrooks.com/tech/unprocessing
# """

# import numpy as np
# import torch
# import torch.distributions as tdist


# def random_ccm(device):
#   """Generates random RGB -> Camera color correction matrices on the given device."""
#   xyz2cams = torch.tensor(
#       [[[1.0234, -0.2969, -0.2266],
#         [-0.5625, 1.6328, -0.0469],
#         [-0.0703, 0.2188, 0.6406]],
#        [[0.4913, -0.0541, -0.0202],
#         [-0.613, 1.3513, 0.2906],
#         [-0.1564, 0.2151, 0.7183]],
#        [[0.838, -0.263, -0.0639],
#         [-0.2887, 1.0725, 0.2496],
#         [-0.0627, 0.1427, 0.5438]],
#        [[0.6596, -0.2079, -0.0562],
#         [-0.4782, 1.3016, 0.1933],
#         [-0.097, 0.1581, 0.5181]]],
#       dtype=torch.float, device=device)
#   num_ccms = xyz2cams.size(0)
#   weights = torch.empty((num_ccms, 1, 1), device=device).uniform_(1e-8, 1e8)
#   weights_sum = torch.sum(weights, dim=0)
#   xyz2cam = torch.sum(xyz2cams * weights, dim=0) / weights_sum

#   # Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
#   rgb2xyz = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
#                            [0.2126729, 0.7151522, 0.0721750],
#                            [0.0193339, 0.1191920, 0.9503041]],
#                           dtype=torch.float, device=device)
#   rgb2cam = torch.mm(xyz2cam, rgb2xyz)

#   # Normalizes each row.
#   rgb2cam = rgb2cam / torch.sum(rgb2cam, dim=-1, keepdim=True)
#   return rgb2cam


# def random_gains(device):
#   """Generates random gains for brightening and white balance on the given device."""
#   # RGB gain represents brightening.
#   n = tdist.Normal(loc=torch.tensor([0.8], device=device),
#                    scale=torch.tensor([0.1], device=device))
#   rgb_gain = 1.0 / n.sample()

#   # Red and blue gains represent white balance.
#   red_gain = torch.empty(1, device=device).uniform_(1.9, 2.4)
#   blue_gain = torch.empty(1, device=device).uniform_(1.5, 1.9)
#   return rgb_gain, red_gain, blue_gain


# def inverse_smoothstep(image):
#   """Approximately inverts a global tone mapping curve."""
#   image = image.permute(1, 2, 0)  # HxWxC
#   image = torch.clamp(image, min=0.0, max=1.0)
#   out = 0.5 - torch.sin(torch.asin(1.0 - 2.0 * image) / 3.0)
#   out = out.permute(2, 0, 1)  # CxHxW
#   return out


# def gamma_expansion(image):
#   """Converts from gamma to linear space."""
#   image = image.permute(1, 2, 0)  # HxWxC
#   out = torch.clamp(image, min=1e-8) ** 2.2
#   out = out.permute(2, 0, 1)  # CxHxW
#   return out


# def apply_ccm(image, ccm):
#   """Applies a color correction matrix."""
#   image = image.permute(1, 2, 0)  # HxWxC
#   shape = image.size()
#   image = torch.reshape(image, [-1, 3])
#   image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
#   out = torch.reshape(image, shape)
#   out = out.permute(2, 0, 1)  # CxHxW
#   return out


# def safe_invert_gains(image, rgb_gain, red_gain, blue_gain):
#   """Inverts gains while safely handling saturated pixels."""
#   device = image.device
#   image = image.permute(1, 2, 0)  # HxWxC
#   gains = torch.stack((1.0 / red_gain,
#                        torch.tensor([1.0], device=device),
#                        1.0 / blue_gain)) / rgb_gain
#   gains = gains.squeeze()
#   gains = gains[None, None, :]
#   # Prevents dimming of saturated pixels by smoothly masking gains near white.
#   gray = torch.mean(image, dim=-1, keepdim=True)
#   inflection = 0.9
#   mask = (torch.clamp(gray - inflection, min=0.0) / (1.0 - inflection)) ** 2.0
#   safe_gains = torch.max(mask + (1.0 - mask) * gains, gains)
#   out = image * safe_gains
#   out = out.permute(2, 0, 1)  # CxHxW
#   return out


# def mosaic(image):
#   """Extracts RGGB Bayer planes from an RGB image."""
#   image = image.permute(1, 2, 0)  # HxWxC
#   shape = image.size()
#   red = image[0::2, 0::2, 0]
#   green_red = image[0::2, 1::2, 1]
#   green_blue = image[1::2, 0::2, 1]
#   blue = image[1::2, 1::2, 2]
#   out = torch.stack((red, green_red, green_blue, blue), dim=-1)
#   out = torch.reshape(out, (shape[0] // 2, shape[1] // 2, 4))
#   out = out.permute(2, 0, 1)  # CxHxW
#   return out


# def unprocess(image):
#   """Unprocesses an image from sRGB to realistic raw data."""
#   device = image.device
#   # 随机生成图像元数据，并确保所有tensor在同一设备上
#   rgb2cam = random_ccm(device)
#   cam2rgb = torch.inverse(rgb2cam)
#   rgb_gain, red_gain, blue_gain = random_gains(device)

#   # 依次反转各个处理步骤
#   image = inverse_smoothstep(image)
#   image = gamma_expansion(image)
#   image = apply_ccm(image, rgb2cam)
#   image = safe_invert_gains(image, rgb_gain, red_gain, blue_gain)
#   image = torch.clamp(image, min=0.0, max=1.0)
#   image = mosaic(image)

#   metadata = {
#       'cam2rgb': cam2rgb,
#       'rgb_gain': rgb_gain,
#       'red_gain': red_gain,
#       'blue_gain': blue_gain,
#   }
#   return image, metadata


# def random_noise_levels(device):
#   """Generates random noise levels from a log-log linear distribution on the given device."""
#   log_min_shot_noise = np.log(0.0001)
#   log_max_shot_noise = np.log(0.012)
#   log_shot_noise = torch.empty(1, device=device).uniform_(log_min_shot_noise, log_max_shot_noise)
#   shot_noise = torch.exp(log_shot_noise)

#   line = lambda x: 2.18 * x + 1.20
#   n = tdist.Normal(loc=torch.tensor([0.0], device=device),
#                    scale=torch.tensor([0.26], device=device))
#   log_read_noise = line(log_shot_noise) + n.sample()
#   read_noise = torch.exp(log_read_noise)
#   return shot_noise, read_noise


# def add_noise(image, shot_noise=0.01, read_noise=0.0005):
#   """Adds random shot (proportional to image) and read (independent) noise."""
#   image = image.permute(1, 2, 0)  # HxWxC
#   variance = image * shot_noise + read_noise
#   n = tdist.Normal(loc=torch.zeros_like(variance), scale=torch.sqrt(variance))
#   noise = n.sample()
#   out = image + noise
#   out = out.permute(2, 0, 1)  # CxHxW
#   return out



# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unprocesses sRGB images into realistic raw data.

Unprocessing Images for Learned Raw Denoising
http://timothybrooks.com/tech/unprocessing
"""

import numpy as np
import torch
import torch.distributions as tdist


def random_ccm(device):
  """Generates random RGB -> Camera color correction matrices on the given device."""
  xyz2cams = torch.tensor(
      [[[1.0234, -0.2969, -0.2266],
        [-0.5625, 1.6328, -0.0469],
        [-0.0703, 0.2188, 0.6406]],
       [[0.4913, -0.0541, -0.0202],
        [-0.613, 1.3513, 0.2906],
        [-0.1564, 0.2151, 0.7183]],
       [[0.838, -0.263, -0.0639],
        [-0.2887, 1.0725, 0.2496],
        [-0.0627, 0.1427, 0.5438]],
       [[0.6596, -0.2079, -0.0562],
        [-0.4782, 1.3016, 0.1933],
        [-0.097, 0.1581, 0.5181]]],
      dtype=torch.float, device=device)
  num_ccms = xyz2cams.size(0)
  weights = torch.empty((num_ccms, 1, 1), device=device).uniform_(1e-8, 1e8)
  weights_sum = torch.sum(weights, dim=0)
  xyz2cam = torch.sum(xyz2cams * weights, dim=0) / weights_sum

  # Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
  rgb2xyz = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                           [0.2126729, 0.7151522, 0.0721750],
                           [0.0193339, 0.1191920, 0.9503041]],
                          dtype=torch.float, device=device)
  rgb2cam = torch.mm(xyz2cam, rgb2xyz)

  # Normalizes each row.
  rgb2cam = rgb2cam / torch.sum(rgb2cam, dim=-1, keepdim=True)
  return rgb2cam


def random_gains(device):
  """Generates random gains for brightening and white balance on the given device."""
  # RGB gain represents brightening.
  n = tdist.Normal(loc=torch.tensor([0.8], device=device),
                   scale=torch.tensor([0.1], device=device))
  rgb_gain = 1.0 / n.sample()

  # Red and blue gains represent white balance.
  red_gain = torch.empty(1, device=device).uniform_(1.9, 2.4)
  blue_gain = torch.empty(1, device=device).uniform_(1.5, 1.9)
  return rgb_gain, red_gain, blue_gain


def inverse_smoothstep(image):
  """Approximately inverts a global tone mapping curve.
     Expects image of shape: (B, C, H, W)
  """
  # 将图像从 (B, C, H, W) 转换为 (B, H, W, C)
  image = image.permute(0, 2, 3, 1)
  image = torch.clamp(image, min=0.0, max=1.0)
  out = 0.5 - torch.sin(torch.asin(1.0 - 2.0 * image) / 3.0)
  out = out.permute(0, 3, 1, 2)  # 转回 (B, C, H, W)
  return out


def gamma_expansion(image):
  """Converts from gamma to linear space.
     Expects image of shape: (B, C, H, W)
  """
  image = image.permute(0, 2, 3, 1)  # (B, H, W, C)
  out = torch.clamp(image, min=1e-8) ** 2.2
  out = out.permute(0, 3, 1, 2)  # (B, C, H, W)
  return out


def apply_ccm(image, ccm):
  """Applies a color correction matrix.
     Expects image of shape: (B, C, H, W) and ccm of shape (3, 3)
  """
  image = image.permute(0, 2, 3, 1)  # (B, H, W, C)
  B, H, W, _ = image.shape
  image = image.reshape(-1, 3)  # (B*H*W, 3)
  # 使用矩阵乘法：image @ ccm.T，得到 (B*H*W, 3)
  out = torch.matmul(image, ccm.t())
  out = out.reshape(B, H, W, 3)
  out = out.permute(0, 3, 1, 2)  # (B, C, H, W)
  return out


def safe_invert_gains(image, rgb_gain, red_gain, blue_gain):
  """Inverts gains while safely handling saturated pixels.
     Expects image of shape: (B, C, H, W)
  """
  device = image.device
  image = image.permute(0, 2, 3, 1)  # (B, H, W, C)
  gains = torch.stack((1.0 / red_gain,
                       torch.tensor([1.0], device=device),
                       1.0 / blue_gain)) / rgb_gain
  gains = gains.squeeze()  # shape (3,)
  gains = gains.view(1, 1, 1, 3)  # 扩展至 (1, 1, 1, 3) 便于广播到 (B, H, W, 3)
  gray = torch.mean(image, dim=-1, keepdim=True)
  inflection = 0.9
  mask = (torch.clamp(gray - inflection, min=0.0) / (1.0 - inflection)) ** 2.0
  safe_gains = torch.max(mask + (1.0 - mask) * gains, gains)
  out = image * safe_gains
  out = out.permute(0, 3, 1, 2)  # (B, C, H, W)
  return out


def mosaic(image):
  """Extracts RGGB Bayer planes from an RGB image.
     Expects image of shape: (B, C, H, W)
  """
  image = image.permute(0, 2, 3, 1)  # (B, H, W, C)
  B, H, W, _ = image.shape
  # 提取 Bayer 格式中的各个通道，每个操作均在 batch 内独立进行
  red = image[:, 0::2, 0::2, 0]         # (B, H/2, W/2)
  green_red = image[:, 0::2, 1::2, 1]     # (B, H/2, W/2)
  green_blue = image[:, 1::2, 0::2, 1]    # (B, H/2, W/2)
  blue = image[:, 1::2, 1::2, 2]          # (B, H/2, W/2)
  out = torch.stack((red, green_red, green_blue, blue), dim=-1)  # (B, H/2, W/2, 4)
  out = out.permute(0, 3, 1, 2)  # (B, 4, H/2, W/2)
  return out


def unprocess(image):
  """Unprocesses an image from sRGB to realistic raw data.
     Expects image of shape: (B, C, H, W)
  """
  device = image.device
  # 随机生成图像元数据，并确保所有 tensor 在同一设备上
  rgb2cam = random_ccm(device)
  cam2rgb = torch.inverse(rgb2cam)
  rgb_gain, red_gain, blue_gain = random_gains(device)

  # 按顺序反转各个处理步骤
  image = inverse_smoothstep(image)
  image = gamma_expansion(image)
  image = apply_ccm(image, rgb2cam)
  image = safe_invert_gains(image, rgb_gain, red_gain, blue_gain)
  image = torch.clamp(image, min=0.0, max=1.0)
  image = mosaic(image)

  metadata = {
      'cam2rgb': cam2rgb,
      'rgb_gain': rgb_gain,
      'red_gain': red_gain,
      'blue_gain': blue_gain,
  }
  return image, metadata


def random_noise_levels(device):
  """Generates random noise levels from a log-log linear distribution on the given device."""
  log_min_shot_noise = np.log(0.0001)
  log_max_shot_noise = np.log(0.012)
  log_shot_noise = torch.empty(1, device=device).uniform_(log_min_shot_noise, log_max_shot_noise)
  shot_noise = torch.exp(log_shot_noise)

  line = lambda x: 2.18 * x + 1.20
  n = tdist.Normal(loc=torch.tensor([0.0], device=device),
                   scale=torch.tensor([0.26], device=device))
  log_read_noise = line(log_shot_noise) + n.sample()
  read_noise = torch.exp(log_read_noise)
  return shot_noise, read_noise


def add_noise(image, shot_noise=0.01, read_noise=0.0005):
  """Adds random shot (proportional to image) and read (independent) noise.
     Expects image of shape: (B, C, H, W)
  """
  image = image.permute(0, 2, 3, 1)  # (B, H, W, C)
  variance = image * shot_noise + read_noise
  n = tdist.Normal(loc=torch.zeros_like(variance), scale=torch.sqrt(variance))
  noise = n.sample()
  out = image + noise
  out = out.permute(0, 3, 1, 2)  # (B, C, H, W)
  return out
