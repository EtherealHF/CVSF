import torch
from torch import nn
import timm
from PIL import Image

# class ImageEncoder(torch.nn.Module):
#     def __init__(self, image_encoder_name, device, preprocess_train=None, vit_model=None):
#         super().__init__()
#         self.device = device
#         self.encoder_name = image_encoder_name.lower()
#         self.preprocess_train = preprocess_train

#         if self.encoder_name == "vit":
#             assert vit_model is not None, "vit_model must be provided when encoder_name=vit"
#             self.model = vit_model
#             self.model.eval()
#             # clip的预处理来自 open_clip
#             self.preprocess = preprocess_train
#         else:
#             # 其它模型来自 timm
#             self.model = timm.create_model(image_encoder_name, pretrained=True, num_classes=0)
#             self.model.eval()
#             self.model = self.model.to(device)
#             # 使用通用预处理
#             self.preprocess = self.default_preprocess

#     def default_preprocess(self, img_path):
#         # timm推荐的标准预处理
#         from torchvision import transforms
#         tfm = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225]),
#         ])
#         img = Image.open(img_path).convert("RGB")
#         return tfm(img)

#     def forward(self, images=None, preprocessed_images=None):
#         batch_size = 20
#         features_list = []

#         if preprocessed_images is not None:
#             assert isinstance(preprocessed_images, torch.Tensor)
#             assert len(preprocessed_images.shape) == 4
#             for i in range(0, len(preprocessed_images), batch_size):
#                 batch_tensor = preprocessed_images[i:i+batch_size].to(self.device)
#                 with torch.no_grad():
#                     if self.encoder_name == "clip":
#                         batch_features = self.model.encode_image(batch_tensor)
#                     else:
#                         batch_features = self.model(batch_tensor)
#                     batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
#                 features_list.append(batch_features)
#             return torch.cat(features_list, dim=0)
#         else:
#             for i in range(0, len(images), batch_size):
#                 batch_images = images[i:i+batch_size]
#                 image_inputs = torch.stack([
#                     self.preprocess(img) if isinstance(img, str) else self.preprocess(img)
#                     for img in batch_images
#                 ]).to(self.device)
#                 with torch.no_grad():
#                     if self.encoder_name == "clip":
#                         batch_features = self.model.encode_image(image_inputs)
#                     else:
#                         batch_features = self.model(image_inputs)
#                     batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
#                 features_list.append(batch_features)
#             return torch.cat(features_list, dim=0)

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
        x = x / x.norm(dim=-1, keepdim=True)  # 归一化（按需）
        return x
