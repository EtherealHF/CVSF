import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage1ImageEncoderWrapper(nn.Module):
    def __init__(self, img_enc: nn.Module, image_size=224,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 freeze=True):
        super().__init__()
        self.img_enc = img_enc
        self.image_size = image_size

        mean = torch.tensor(mean).view(1, 3, 1, 1)
        std = torch.tensor(std).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        if freeze:
            self.img_enc.eval()
            for p in self.img_enc.parameters():
                p.requires_grad = False

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # x: [-1, 1] -> [0, 1]
        x = (x + 1.0) / 2.0
        x = x.clamp(0, 1)

        x = F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        x = (x - self.mean) / self.std
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        feat = self.img_enc(x)
        return feat


def load_stage1_img_enc(build_img_enc_fn, ckpt_path, device="cpu", strict=True):
    img_enc = build_img_enc_fn()
    ckpt = torch.load(ckpt_path, map_location=device)

    if "img_enc" in ckpt:
        state_dict = ckpt["img_enc"]
    elif "model" in ckpt and "img_enc" in ckpt["model"]:
        state_dict = ckpt["model"]["img_enc"]
    elif "state_dict_img_enc" in ckpt:
        state_dict = ckpt["state_dict_img_enc"]
    else:
        # 兼容 alignment_train.py directly saved Stage-1 image encoder checkpoint
        state_dict = ckpt

    img_enc.load_state_dict(state_dict, strict=strict)
    return img_enc

