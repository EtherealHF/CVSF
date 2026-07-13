import torch
from torch import nn
import timm

class ImageEncoder(nn.Module):
    def __init__(self, backbone='efficientnet_b0', out_dim=1024, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feature_dim = self.backbone.num_features
        self.project = nn.Linear(feature_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        x = self.backbone(x)
        x = self.project(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x
