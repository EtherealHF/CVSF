import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import numpy as np
import re
from einops.layers.torch import Rearrange
from .subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from .subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from .subject_layers.Embed import DataEmbedding
from .utils.loss import ClipLoss
from torch import Tensor

# class Config:
#     def __init__(self):
#         self.task_name = 'classification'  # Example task name
#         self.seq_len = 250                 # Sequence length
#         self.pred_len = 250                # Prediction length
#         self.output_attention = False      # Whether to output attention weights
#         self.d_model = 250                 # Model dimension
#         self.embed = 'timeF'               # Time encoding method
#         self.freq = 'h'                    # Time frequency
#         self.dropout = 0.25                # Dropout rate
#         self.factor = 1                    # Attention scaling factor
#         self.n_heads = 4                   # Number of attention heads
#         self.e_layers = 1                  # Number of encoder layers
#         self.d_ff = 256                    # Dimension of the feedforward network
#         self.activation = 'gelu'           # Activation function
#         self.enc_in = 63                   # Encoder input dimension (example value)
        
# class iTransformer(nn.Module):
#     def __init__(self, configs, joint_train=False,  num_subjects=10):
#         super(iTransformer, self).__init__()
#         self.task_name = configs.task_name
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.output_attention = configs.output_attention
#         # Embedding
#         self.enc_embedding = DataEmbedding(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout, joint_train=False, num_subjects=num_subjects)
#         # Encoder
#         self.encoder = Encoder(
#             [
#                 EncoderLayer(
#                     AttentionLayer(
#                         FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
#                         configs.d_model, configs.n_heads
#                     ),
#                     configs.d_model,
#                     configs.d_ff,
#                     dropout=configs.dropout,
#                     activation=configs.activation
#                 ) for l in range(configs.e_layers)
#             ],
#             norm_layer=torch.nn.LayerNorm(configs.d_model)
#         )

#     def forward(self, x_enc, x_mark_enc, subject_ids=None):
#         # Embedding
#         enc_out = self.enc_embedding(x_enc, x_mark_enc, subject_ids)
#         enc_out, attns = self.encoder(enc_out, attn_mask=None)
#         enc_out = enc_out[:, :63, :]      
#         # print("enc_out", enc_out.shape)
#         return enc_out

# class PatchEmbedding(nn.Module):
#     def __init__(self, emb_size=40):
#         super().__init__()
#         # Revised from ShallowNet
#         self.tsconv = nn.Sequential(
#             nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
#             nn.AvgPool2d((1, 51), (1, 5)),
#             nn.BatchNorm2d(40),
#             nn.ELU(),
#             # nn.Conv2d(40, 40, (63, 1), stride=(1, 1)),
#             nn.Conv2d(40, 40, (18, 1), stride=(1, 1)),
#             nn.BatchNorm2d(40),
#             nn.ELU(),
#             nn.Dropout(0.5),
#         )

#         self.projection = nn.Sequential(
#             nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
#             Rearrange('b e (h) (w) -> b (h w) e'),
#         )

#     def forward(self, x: Tensor) -> Tensor:
#         # b, _, _, _ = x.shape
#         x = x.unsqueeze(1)     
#         # print("x", x.shape)   
#         x = self.tsconv(x)
#         # print("tsconv", x.shape)   
#         x = self.projection(x)
#         # print("projection", x.shape)  
#         return x

# class ResidualAdd(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn

#     def forward(self, x, **kwargs):
#         res = x
#         x = self.fn(x, **kwargs)
#         x += res
#         return x

# class FlattenHead(nn.Sequential):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         x = x.contiguous().view(x.size(0), -1)
#         return x

# class Enc_eeg(nn.Sequential):
#     def __init__(self, emb_size=40, **kwargs):
#         super().__init__(
#             PatchEmbedding(emb_size),
#             FlattenHead()
#         )

# class Proj_eeg(nn.Sequential):
#     def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
#         super().__init__(
#             nn.Linear(embedding_dim, proj_dim),
#             ResidualAdd(nn.Sequential(
#                 nn.GELU(),
#                 nn.Linear(proj_dim, proj_dim),
#                 nn.Dropout(drop_proj),
#             )),
#             nn.LayerNorm(proj_dim),
#         )

# class ATMS(nn.Module):    
#     def __init__(self, num_channels=63, sequence_length=250, num_subjects=2, num_features=64, num_latents=1024, num_blocks=1):
#         super(ATMS, self).__init__()
#         default_config = Config()
#         self.encoder = iTransformer(default_config)   
#         self.subject_wise_linear = nn.ModuleList([nn.Linear(default_config.d_model, sequence_length) for _ in range(num_subjects)])
#         self.enc_eeg = Enc_eeg()
#         self.proj_eeg = Proj_eeg()        
#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
#         self.loss_func = ClipLoss()       
         
#     def forward(self, x, subject_ids):
#         x = self.encoder(x, None, subject_ids)
#         # print(f"x.shape: {x.shape}")

#         eeg_embedding = self.enc_eeg(x)
#         # print(f"eeg_embedding shape: {eeg_embedding.shape}")
        
#         out = self.proj_eeg(eeg_embedding)
#         # print(f"out.shape: {out.shape}")
#         return out 
class Config:
    def __init__(self):
        self.task_name = 'classification'  # Example task name
        self.seq_len = 250                 # Sequence length
        self.pred_len = 250                # Prediction length
        self.output_attention = False      # Whether to output attention weights
        self.d_model = 250                 # Model dimension
        self.embed = 'timeF'               # Time encoding method
        self.freq = 'h'                    # Time frequency
        self.dropout = 0.25                # Dropout rate
        self.factor = 1                    # Attention scaling factor
        self.n_heads = 4                   # Number of attention heads
        self.e_layers = 1                  # Number of encoder layers
        self.d_ff = 256                    # Dimension of the feedforward network
        self.activation = 'gelu'           # Activation function
        self.enc_in = 63                   # Encoder input dimension (example value)

class iTransformer(nn.Module):
    def __init__(self, configs, joint_train=False,  num_subjects=10):
        super(iTransformer, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout, joint_train=False, num_subjects=num_subjects)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

    def forward(self, x_enc, x_mark_enc, subject_ids=None):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc, subject_ids)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = enc_out[:, :63, :]      
        # print("enc_out", enc_out.shape)
        return enc_out

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # Revised from ShallowNet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (18, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        # print("x", x.shape)   
        x = x.unsqueeze(1)
        # x: [64, 1, 18, 250]
        # print("x", x.shape)
        x = self.tsconv(x)
        # print("tsconv", x.shape)   
        x = self.projection(x)
        # print("projection", x.shape)
        return x

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x

class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )

class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

class ATMS(nn.Module):    
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=2, num_features=64, num_latents=1024, num_blocks=1):
        super(ATMS, self).__init__()
        default_config = Config()
        self.encoder = iTransformer(default_config)   
        self.subject_wise_linear = nn.ModuleList([nn.Linear(default_config.d_model, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()       
         
    def forward(self, x, subject_ids):
        x = self.encoder(x, None, subject_ids)
        # print(f'After attention shape: {x.shape}')
        # print("x", x.shape) # x: [64,18,250]
        # x = self.subject_wise_linear[0](x)
        # print(f'After subject-specific linear transformation shape: {x.shape}')
        eeg_embedding = self.enc_eeg(x)
        
        out = self.proj_eeg(eeg_embedding)
        return out  

def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None

import torch
import numpy as np

def seed_torch(seed=1029):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    # 设置随机种子
    seed_torch(7)

    # 设置参数
    batch_size = 32
    num_channels = 63
    sequence_length = 250
    num_subjects = 2
    num_features = 64
    num_latents = 1024
    num_blocks = 1

    # 初始化模型
    model = ATMS(
        num_channels=num_channels,
        sequence_length=sequence_length,
        num_subjects=num_subjects,
        num_features=num_features,
        num_latents=num_latents,
        num_blocks=num_blocks
    )

    # 随机生成测试数据
    eeg_data = torch.randn(batch_size, num_channels, sequence_length)  # [B, 63, 250]
    subject_ids = torch.randint(0, num_subjects, (batch_size,))       # 随机生成 subject IDs

    # 模型前向传播
    model.eval()
    with torch.no_grad():
        output = model(eeg_data, subject_ids)

    # 验证输出形状
    expected_shape = (batch_size, num_latents)
    # print(f"Input shape: {eeg_data.shape}")
    # print(f"Subject IDs shape: {subject_ids.shape}")
    # print(f"Output shape: {output.shape}")
    # if output.shape == expected_shape:
    #     print("Test passed: ATMS model correctly transforms [B, 63, 250] to [B, 1024]")
    # else:
    #     print(f"Test failed: Expected output shape {expected_shape}, but got {output.shape}")

if __name__ == "__main__":
    main()