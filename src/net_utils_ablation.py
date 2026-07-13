import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
from einops import rearrange

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class SelfAttention(nn.Module):
    def __init__(self, inp_channel):
        super().__init__()
        self.inp_channel=inp_channel
        self.norm=LayerNorm(inp_channel)
        self.conv1x1=nn.Conv2d(inp_channel,inp_channel,kernel_size=1)
        self.DWConv=nn.Conv2d(inp_channel,inp_channel,kernel_size=3, padding=1, groups=inp_channel)
        self.adaptive_dconv=nn.Conv2d(inp_channel,inp_channel, kernel_size=3, padding=1, groups=inp_channel)
        self.adaptive_liner=nn.Conv2d(inp_channel, inp_channel*2,kernel_size=1)
        self.out=nn.Conv2d(inp_channel,inp_channel,kernel_size=1)
    def forward(self,x):
        inp=x
        x = self.norm(x)
        q, k = self.adaptive_liner(self.adaptive_dconv(x)).chunk(2,dim=1)
        v = self.DWConv(self.conv1x1(x))
        q=rearrange(q,'B C H W -> B C (H W)')
        k=rearrange(k,'B C H W -> B C (H W)')
        attn=(q@k.transpose(-2,-1)).softmax(dim=-1)
        B,C,H,W=x.shape
        v=rearrange(v,'B C H W -> B C (H W)')
        x=rearrange(attn@v, 'B C (H W) -> B C H W',H=H,W=W)
        x=self.out(x)
        x=x+inp
        return x


# class CrossAttention(nn.Module):
#     def __init__(self, inp_channel):
#         super().__init__()
#         self.inp_channel = inp_channel
#         self.norm = LayerNorm(inp_channel)
#         self.conv1x1 = nn.Conv2d(inp_channel, inp_channel, kernel_size=1)
#         self.DWConv = nn.Conv2d(inp_channel, inp_channel,
#                                 kernel_size=3, padding=1, groups=inp_channel)
#         self.adaptive_dconv = nn.Conv2d(
#             inp_channel, inp_channel, kernel_size=3, padding=1, groups=inp_channel)
#         self.adaptive_liner = nn.Conv2d(
#             inp_channel, inp_channel*2, kernel_size=1)
#         self.out = nn.Conv2d(inp_channel, inp_channel, kernel_size=1)

#     def forward(self, x):
#         inp = x
#         x = self.norm(x)
#         q, k = self.adaptive_liner(self.adaptive_dconv(x)).chunk(2, dim=1)
#         v = self.DWConv(self.conv1x1(x))
#         q = rearrange(q, 'B C H W -> B C (H W)')
#         k = rearrange(k, 'B C H W -> B C (H W)')
#         attn = (q@k.transpose(-2, -1)).softmax(dim=-1)
#         B, C, H, W = x.shape
#         v = rearrange(v, 'B C H W -> B C (H W)')
#         x = rearrange(attn@v, 'B C (H W) -> B C H W', H=H, W=W)
#         x = self.out(x)
#         x = x+inp
#         return x


class CrossAttention(nn.Module):
    def __init__(self, inp_channel):
        super().__init__()
        self.inp_channel = inp_channel
        self.norm_q = LayerNorm(inp_channel)
        self.adaptive_dconv_q = nn.Conv2d(
            inp_channel, inp_channel, kernel_size=3, padding=1, groups=inp_channel)
        self.adaptive_linear_q = nn.Conv2d(
            inp_channel, inp_channel, kernel_size=1)
        self.norm_kv = LayerNorm(3 * inp_channel)
        self.adaptive_dconv_kv = nn.Conv2d(
            3 * inp_channel, 3 * inp_channel, kernel_size=3, padding=1, groups=3 * inp_channel)
        self.adaptive_linear_kv = nn.Conv2d(
            3 * inp_channel, 2 * inp_channel, kernel_size=1)
        self.out = nn.Conv2d(inp_channel, inp_channel, kernel_size=1)

    def forward(self, x_q, x_kv):
        inp = x_q
        x_q = self.norm_q(x_q)
        q = self.adaptive_linear_q(self.adaptive_dconv_q(x_q))  # (B, C, H, W)
        x_kv = self.norm_kv(x_kv)
        kv = self.adaptive_linear_kv(
            self.adaptive_dconv_kv(x_kv))  # (B, 2C, H, W)
        k, v = kv.chunk(2, dim=1)  # K: (B, C, H, W), V: (B, C, H, W)
        B, C, H, W = q.shape
        q = rearrange(q, 'B C H W -> B C (H W)')
        k = rearrange(k, 'B C H W -> B C (H W)')
        v = rearrange(v, 'B C H W -> B C (H W)')
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        x = attn @ v  # (B, C, H*W)
        x = rearrange(x, 'B C (H W) -> B C H W', H=H, W=W)
        x = self.out(x)
        x = x + inp
        return x

class WaveletPriorExtraction(nn.Module):
    def __init__(self, in_channels=3, feature_channels=256, down_size=8):
        super().__init__()
        
        self.down_size = down_size
        self.down = nn.PixelUnshuffle(self.down_size)
        
        self.conv_in = nn.Conv2d(in_channels*self.down_size*self.down_size, feature_channels, kernel_size=3, padding=1)
        
        # self.dwt = DWTForward(J=1, wave='haar')
        
        # self.low_freq_proc = CrossAttention(feature_channels)

        
    def forward(self, x):
        """
        杈撳叆 x: [B, in_channels, H, W]
        """
        x = self.down(x)
        feat = self.conv_in(x)  # [B, feature_channels, H, W]
        feat_reconstructed = self.high_freq_proc(feat)  # [B, feature_channels, H, W]
        
        # Yl, Yh = self.dwt(feat)
        
        # Yh0_flat = Yh[0].reshape(B, C * num_subbands, H2, W2)  # [B, 3*feature_channels, H/2, W/2]
    
        # low_processed = self.low_freq_proc(Yl, Yh0_flat)
        # high_processed_flat = self.high_freq_proc(Yh0_flat)
    
        
        # feat_reconstructed = self.idwt((low_processed, high_processed))
        
        
        return out




class DWTPostProcess(nn.Module):
    def __init__(self, in_channels=3, feature_channels=256):
        super().__init__()

        
        self.conv_in = nn.Conv2d(in_channels*2, feature_channels, kernel_size=3, padding=1)
        
        self.dwt = DWTForward(J=1, wave='haar')
        
        self.low_freq_proc = CrossAttention(feature_channels)



        self.final_proc = SelfAttention(feature_channels)


        
    def forward(self, x_lq, x_en):
        """
        杈撳叆 x: [B, in_channels, H, W]
        """
        # x = self.down(x)
        x = torch.cat([x_lq, x_en], dim=1)
        feat = self.conv_in(x)  # [B, feature_channels, H, W]
        
        Yl, Yh = self.dwt(feat)
        
        B, C, num_subbands, H2, W2 = Yh[0].shape  # C 绛変簬 feature_channels, num_subbands 搴斾负 3
        Yh0_flat = Yh[0].reshape(B, C * num_subbands, H2, W2)  # [B, 3*feature_channels, H/2, W/2]
    
        low_processed = self.low_freq_proc(Yl, Yh0_flat)
        high_processed_flat = self.high_freq_proc(Yh0_flat)
    
        
        feat_reconstructed = self.idwt((low_processed, high_processed))

        out = self.final_proc(feat_reconstructed)
        
        
        return out
    
    def save_model(self, outf):
        torch.save(self.state_dict, outf)


    def set_train(self):
        self.train()
        self.requires_grad_(True)

    def set_eval(self):
        self.eval()
        self.requires_grad_(False)


class CrossAttention_1(nn.Module):
    def __init__(self, inp_channel):
        super().__init__()
        self.inp_channel = inp_channel
        self.norm_q = LayerNorm(inp_channel)
        self.adaptive_dconv_q = nn.Conv2d(
            inp_channel, inp_channel, kernel_size=3, padding=1, groups=inp_channel)
        self.adaptive_linear_q = nn.Conv2d(
            inp_channel, inp_channel, kernel_size=1)
        self.norm_kv = LayerNorm(inp_channel)
        self.adaptive_dconv_kv = nn.Conv2d(
            inp_channel, inp_channel, kernel_size=3, padding=1, groups=inp_channel)
        self.adaptive_linear_kv = nn.Conv2d(
            inp_channel, 2 * inp_channel, kernel_size=1)
        self.out = nn.Conv2d(inp_channel, inp_channel, kernel_size=1)

    def forward(self, x_q, x_kv):
        inp = x_q
        x_q = self.norm_q(x_q)
        q = self.adaptive_linear_q(self.adaptive_dconv_q(x_q))  # (B, C, H, W)
        x_kv = self.norm_kv(x_kv)
        kv = self.adaptive_linear_kv(
            self.adaptive_dconv_kv(x_kv))  # (B, 2C, H, W)
        k, v = kv.chunk(2, dim=1)  # K: (B, C, H, W), V: (B, C, H, W)
        B, C, H, W = q.shape
        q = rearrange(q, 'B C H W -> B C (H W)')
        k = rearrange(k, 'B C H W -> B C (H W)')
        v = rearrange(v, 'B C H W -> B C (H W)')
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        x = attn @ v  # (B, C, H*W)
        x = rearrange(x, 'B C (H W) -> B C H W', H=H, W=W)
        x = self.out(x)
        x = x + inp
        return x


class DWTPostProcess1(nn.Module):
    def __init__(self, in_channels=3, feature_channels=32):
        super().__init__()

        
        self.conv_in_lq = nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1)
        self.conv_in_en = nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1)
        
        self.dwt = DWTForward(J=1, wave='haar')
        
        self.low_freq_proc = CrossAttention_1(feature_channels)
        
        # self.high_freq_proc = SelfAttention(feature_channels * 3)



        self.final_proc = SelfAttention(feature_channels)


        
    def forward(self, x_lq, x_en):
        """
        杈撳叆 x: [B, in_channels, H, W]
        """
        # x = self.down(x)
        # x = torch.cat([x_lq, x_en], dim=1)

        fea_lq = self.conv_in_lq(x_lq)
        fea_en = self.conv_in_lq(x_en)

        # feat = self.conv_in(x)  # [B, feature_channels, H, W]
        
        Yl, Yh = self.dwt(fea_lq)
        B, C, num_subbands, H2, W2 = Yh[0].shape  # C 绛変簬 feature_channels, num_subbands 搴斾负 3
        Yh0_flat = Yh[0].reshape(B, C * num_subbands, H2, W2)  # [B, 3*feature_channels, H/2, W/2]
    
        Yl_, Yh_ = self.dwt(fea_en)
        Yh0_flat_ = Yh[0].reshape(B, C * num_subbands, H2, W2)  # [B, 3*feature_channels, H/2, W/2]
    

        low_processed = self.low_freq_proc(Yl, Yl_)
        high_processed_flat = self.high_freq_proc(Yh0_flat, Yh0_flat_)
    
        
        feat_reconstructed = self.idwt((low_processed, high_processed))

        out = self.final_proc(feat_reconstructed)
        
        
        return out
    
    def save_model(self, outf):
        torch.save(self.state_dict, outf)


    def set_train(self):
        self.train()
        self.requires_grad_(True)

    def set_eval(self):
        self.eval()
        self.requires_grad_(False)
