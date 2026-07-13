import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
import itertools
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from types import SimpleNamespace
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
from model.ImageEncoder import ImageEncoder
# from model.EEG_Encoder import TSLANet, EEGTransformer, ATMS, BENDR, ACNet, FBCNet, EEGNet, Conformer, NICE,  ETNet,  ConvRNN
from model.EEG_Encoder import ATMS
from utils_loc.loss import ClipLoss

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def func_attention(query, context, opt, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    device = query.device if query.is_cuda else context.device
    query = query.to(device)
    context = context.to(device)

    queryT = torch.transpose(query, 1, 2).to(device)  # 显式指定设备


    # Get attention
    # --> (batch, d, queryL)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    if opt.raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size*sourceL, queryL)
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif opt.raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "l1norm":
        attn = l1norm_d(attn, 2)
    elif opt.raw_feature_norm == "clipped_l1norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l1norm_d(attn, 2)
    elif opt.raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif opt.raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", opt.raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)
    attn = nn.Softmax()(attn*smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    # weightedContext: 加权后的上下文表示, 维度为(batch_size, queryL, d)
    # attnT: 转置之后的注意力矩阵, 维度为(batch_size, sourceL, queryL)
    return weightedContext, attnT

def xattn_score_i2e(images, eeg_data, opt):
    """
    Images: (n_image, n_subimage, d) matrix of images
    EEG Data: (n_eeg, n_subeeg, d) matrix of EEG signals
    """
    similarities = []
    n_image = images.size(0)
    n_eeg = eeg_data.size(0)
    
    for i in range(n_eeg):
        # Get the i-th EEG signal (n_subeeg, d)
        n_subeeg = eeg_data.size(1)
        eeg_i = eeg_data[i, :n_subeeg, :].unsqueeze(0).contiguous()
        # --> (n_image, n_subeeg, d)
        eeg_i_expand = eeg_i.repeat(n_image, 1, 1)
        
        """
            eeg(query): (n_image, n_subeeg, d)
            image(context): (n_image, n_subimage, d)
            weiContext: (n_image, n_subeeg, d)
            attn: (n_image, n_subimage, n_subeeg)
        """
        weiContext, attn = func_attention(eeg_i_expand, images, opt, smooth=opt.lambda_softmax)
        
        eeg_i_expand = eeg_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        
        # (n_image, n_subeeg)
        row_sim = cosine_similarity(eeg_i_expand, weiContext, dim=2)
        
        # 聚合方式
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim) / opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        
        similarities.append(row_sim)

    # (n_image, n_eeg)
    similarities = torch.cat(similarities, 1)
    
    return similarities

def xattn_score_e2i(eeg_data, images, opt):
    """
    EEG Data: (n_eeg, n_bands, d) matrix of EEG features
    Images: (n_image, n_regions, d) matrix of images (如4个图片分量)
    img_lens: (n_image) array of图片区域数量（如果每张都一样，可不用）
    """
    similarities = []
    n_eeg = eeg_data.size(0)
    n_image = images.size(0)
    n_region = images.size(1)  # 每张图片的区域/分量数

    for i in range(n_image):
        # 取第i张图片的所有区域/分量 (n_region, d)
        n_subimg = n_region  # 若每张图片区域数一样
        img_i = images[i, :n_subimg, :].unsqueeze(0).contiguous()
        # --> (n_eeg, n_region, d)
        img_i_expand = img_i.repeat(n_eeg, 1, 1)
        """
            image(query): (n_eeg, n_region, d)
            eeg(context): (n_eeg, n_bands, d)
            weiContext: (n_eeg, n_region, d)
            attn: (n_eeg, n_bands, n_region)
        """
        weiContext, attn = func_attention(img_i_expand, eeg_data, opt, smooth=opt.lambda_softmax)
        img_i_expand = img_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        # (n_eeg, n_region)
        row_sim = cosine_similarity(img_i_expand, weiContext, dim=2)
        # 聚合方式
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)/opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)

    # (n_eeg, n_image)
    similarities = torch.cat(similarities, 1)
    return similarities

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss for cross-modal matching (e.g. EEG-Img)
    """
    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, images, eeg):
        """
        images: (n_image, n_region, d)
        eeg: (n_eeg, n_band, d)
        """
        cross_attn = self.opt.cross_attn

        if cross_attn == 'eeg2img':
            # EEG为主轴检索图片，输出(n_eeg, n_image)
            scores = xattn_score_e2i(eeg, images, self.opt)  # 注意参数顺序！
            diagonal = scores.diag().view(eeg.size(0), 1)        # (n_eeg, 1)
            d1 = diagonal.expand_as(scores)                      # (n_eeg, n_image)
            d2 = diagonal.t().expand_as(scores)                  # (n_image, n_eeg)
        elif cross_attn == 'img2eeg':
            # 图片为主轴检索EEG，输出(n_image, n_eeg)
            scores = xattn_score_i2e(images, eeg, self.opt)
            diagonal = scores.diag().view(images.size(0), 1)     # (n_image, 1)
            d1 = diagonal.expand_as(scores)                      # (n_image, n_eeg)
            d2 = diagonal.t().expand_as(scores)                  # (n_eeg, n_image)
        else:
            raise ValueError("Unknown cross_attn type: {}".format(cross_attn))

        # 行损失：每个“主模态”检索其它副模态
        cost_1 = (self.margin + scores - d1).clamp(min=0)
        # 列损失：每个“副模态”检索主模态
        cost_2 = (self.margin + scores - d2).clamp(min=0)

        # 清空对角线（正例损失设为0）
        mask = torch.eye(scores.size(0)) > 0.5 if cross_attn == 'eeg2img' else torch.eye(scores.size(1)) > 0.5
        mask = mask.to(scores.device)
        if cross_attn == 'eeg2img':
            cost_1 = cost_1.masked_fill_(mask, 0)
            cost_2 = cost_2.masked_fill_(mask.t(), 0)
        else:  # img2eeg
            cost_1 = cost_1.masked_fill_(mask, 0)
            cost_2 = cost_2.masked_fill_(mask.t(), 0)

        # max_violation: 只选最大负例
        if self.max_violation:
            cost_1 = cost_1.max(1)[0]
            cost_2 = cost_2.max(0)[0]

        return cost_1.sum() + cost_2.sum()
    
class LocalLoss(nn.Module):
    def __init__(self, opt, clip_loss_func, logit_scale):
        super(LocalLoss, self).__init__()
        self.opt = opt
        self.clip_loss_func = clip_loss_func
        self.logit_scale = logit_scale
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) 
    
    def forward(self, images, eeg):
        """
        images: (n_image, n_region, d)    high, low, have_color, no_color
        eeg: (n_eeg, n_band, d)           beta, gamma
        """

        image_high, image_low = images[:, 0, :], images[:, 1, :]
        eeg_beta, eeg_gamma = eeg[:, 0, :], eeg[:, 1, :]

        loss_high_beta = self.clip_loss_func(image_high, eeg_beta, self.logit_scale)
        loss_high_gamma = self.clip_loss_func(image_high, eeg_gamma, self.logit_scale)
        loss_low_beta = self.clip_loss_func(image_low, eeg_beta, self.logit_scale)
        loss_low_gamma = self.clip_loss_func(image_low, eeg_gamma, self.logit_scale)

        return loss_high_beta, loss_high_gamma, loss_low_beta, loss_low_gamma


class SCAN(object):
    def __init__(self, opt):
        # 保留原有超参
        self.args = SimpleNamespace(
            seq_len=250,
            batch_size=32,
            seed=42,
            emb_dim=opt.emb_dim,
            depth=3,
            dropout_rate=0.5,
            patch_size=50,
            mask_ratio=0.4,
            ICB=True,
            ASB=True,
            adaptive_filter=True,
            num_channels=17,
            out_dim=1024, 
        ) 
        self.device = opt.device
        # 编码器：图像和EEG
        self.img_enc = ImageEncoder(
            backbone=opt.backbone,
            out_dim=opt.embed_size,
            pretrained=True
        ).to(self.device) 
        self.opt = opt
        # self.eeg_enc = TSLANet(self.args)# EEGTransformer()  # 或 TSLANet，根据实际使用
        # if opt.eeg_encoder_type == "EEGTransformer":
        #     self.eeg_enc = globals()[opt.eeg_encoder_type]()
        # elif opt.eeg_encoder_type == "TSLANet":
        #     self.eeg_enc = globals()[opt.eeg_encoder_type](self.args)
        # elif opt.eeg_encoder_type == "ATMS":
        #     self.eeg_enc = globals()[opt.eeg_encoder_type]()
        # elif opt.eeg_encoder_type == "BIOTEncoder":
        #     self.eeg_enc = globals()[opt.eeg_encoder_type]()
        # elif opt.eeg_encoder_type == "BENDR":
        #     self.eeg_enc = globals()[opt.eeg_encoder_type](
        #         channels=self.args.num_channels,
        #         global_dim=self.args.out_dim  # 设置 global_dim=1024
        #     )
        # elif opt.eeg_encoder_type == "ACNet":
        #     self.eeg_enc = globals()[opt.eeg_encoder_type]()
        # elif opt.eeg_encoder_type == "FBCNet":
        #     self.eeg_enc = globals()[opt.eeg_encoder_type](n_chans=17, n_times=250, n_outputs=10, sfreq=250)
        # elif opt.eeg_encoder_type == "EEGNet":
        #     self.eeg_enc = globals()[opt.eeg_encoder_type]()
        # elif opt.eeg_encoder_type == "Conformer":
        #     self.eeg_enc = globals()[opt.eeg_encoder_type]()
        # elif opt.eeg_encoder_type == "NICE":
        #     self.eeg_enc = globals()[opt.eeg_encoder_type]()
        # elif opt.eeg_encoder_type == "ConvRNN":
        #     self.eeg_enc = globals()[opt.eeg_encoder_type]()
        
        # self.eeg_enc = self.eeg_enc.to(self.device)
        if opt.eeg_encoder_type == "ATMS":
            self.eeg_enc = ATMS().to(self.device)
        # if opt.eeg_encoder_type == "EEGTransformer":
        #     img_size = (opt.num_channels, opt.sequence_length)
        #     self.eeg_enc = EEGTransformer(img_size=img_size).to(self.device)
        # elif opt.eeg_encoder_type == "TSLANet":
        #     self.eeg_enc = TSLANet(self.args).to(self.device)
        # elif opt.eeg_encoder_type == "ATMS":
        #     self.eeg_enc = ATMS().to(self.device)
        # elif opt.eeg_encoder_type == "BIOTEncoder":
        #     self.eeg_enc = BIOTEncoder().to(self.device)
        # elif opt.eeg_encoder_type == "BENDR":
        #     self.eeg_enc = BENDR(
        #         channels=self.args.num_channels,
        #         global_dim=self.args.out_dim
        #     ).to(self.device)
        # elif opt.eeg_encoder_type == "ACNet":
        #     self.eeg_enc = ACNet().to(self.device)
        # elif opt.eeg_encoder_type == "FBCNet":
        #     self.eeg_enc = FBCNet(n_chans=17, n_times=250, n_outputs=10, sfreq=250).to(self.device)
        # elif opt.eeg_encoder_type == "EEGNet":
        #     self.eeg_enc = EEGNet().to(self.device)
        # elif opt.eeg_encoder_type == "Conformer":
        #     self.eeg_enc = Conformer().to(self.device)
        # elif opt.eeg_encoder_type == "NICE":
        #     self.eeg_enc = NICE().to(self.device)
        # elif opt.eeg_encoder_type == "ConvRNN":
        #     self.eeg_enc = ConvRNN(
        #         in_channels={"meg": opt.num_channels},  # num_channels=17
        #         out_channels=opt.embed_size,  # embed_size=1024
        #         hidden={"meg": 256},
        #         n_subjects=opt.num_subjects,  # num_subjects=2
        #         depth=2,
        #         linear_out=True,
        #         subject_dim=64,
        #         lstm=2,
        #         attention=0,
        #         kernel_size=4,
        #         stride=2
        #     ).to(self.device)
        # elif opt.eeg_encoder_type == "ETNet":
        #     args = SimpleNamespace(
        #         seq_len=250,
        #         batch_size=32,
        #         emb_dim=64,
        #         depth=3,
        #         dropout_rate=0.5,
        #         patch_size=10,
        #         out_dim=1024,
        #         num_channels=self.opt.num_channels,
        #         top_k=2,
        #         d_ff=128,
        #         num_kernels=6,
        #         TB=True,
        #         ASB=True,
        #         adaptive_filter=True
        #     )
        #     self.eeg_enc = ETNet(args).to(self.device)
        # self.eeg_enc = EEGTransformer()
        if opt.num_channels == 17:
            use_channel_names = ['O1', 'OZ', 'O2', 'PO7', 'PO3', 'POZ','PO4', 'PO8', 'P7', 'P5',
                'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'] 
        else: 
            use_channel_names = ['FP1', 'FP2', 'AF7', 'AF3', 'AFZ', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                                         'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
                                         'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                                         'CZ', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
                                         'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                                         'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8',
                                         'O1', 'OZ', 'O2']
        if opt.eeg_encoder_type == "EEGTransformer":
            self.chan_ids = self.eeg_enc.prepare_chan_ids(use_channel_names)
            self.chan_ids = self.chan_ids.to(self.device)
        self.logit_scale = nn.Parameter(torch.ones([], device=self.device) * np.log(1 / 0.07))
        
        if torch.cuda.is_available():
            # self.img_enc.cuda()
            # self.eeg_enc.cuda()
            cudnn.benchmark = True
        
        self.clip_loss_func = ClipLoss()

        # 损失和优化器
        # self.criterion = ContrastiveLoss(opt=opt,
        #                                  margin=opt.margin,
        #                                  max_violation=opt.max_violation)
        self.criterion = LocalLoss(opt=opt, clip_loss_func=self.clip_loss_func, logit_scale=self.logit_scale) 
        params = itertools.chain(self.img_enc.parameters(), self.eeg_enc.parameters(),[self.logit_scale])

        self.params = params
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.Eiters = 0
        
        # 添加全局损失权重参数
        self.lambda_global = opt.lambda_global
        self.lambda_high_gamma = opt.lambda_high_gamma
        self.lambda_high_beta = opt.lambda_high_beta
        self.lambda_low_gamma = opt.lambda_low_gamma
        self.lambda_low_beta = opt.lambda_low_beta

    def state_dict(self):
        return {
            "img_enc": self.img_enc.state_dict(),
            "eeg_enc": self.eeg_enc.state_dict(),
            "logit_scale": self.logit_scale.data.clone()  # 或直接 .state_dict()，如果是nn.Parameter
        }

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict["img_enc"])
        self.eeg_enc.load_state_dict(state_dict["eeg_enc"])
        self.logit_scale.data.copy_(state_dict["logit_scale"])

    def train_start(self):
        self.img_enc.train()
        self.eeg_enc.train()

    def val_start(self):
        self.img_enc.eval()
        self.eeg_enc.eval()
    
    def parameters(self):
        return self.params

    def forward_emb(self, images, eeg_data, raw_images, raw_eeg_data, volatile=False):
        """
        修正后的编码逻辑，确保输出形状符合要求
        images:    (batch_size, n_regions, 3, 224, 224)
        eeg_data:  (batch_size, n_band, C, T)
        raw_images: (batch_size, 3, 224, 224)
        raw_eeg_data: (batch_size, C, T)
        """
        images = images.to(self.device)
        eeg_data = eeg_data.to(self.device)
        raw_images = raw_images.to(self.device)
        raw_eeg_data = raw_eeg_data.to(self.device)
        images = Variable(images, volatile=volatile)
        eeg_data = Variable(eeg_data, volatile=volatile)
        if torch.cuda.is_available():
            images = images.to(self.device)
            eeg_data = eeg_data.to(self.device)

        # 图像编码处理
        batch_size, n_regions = images.shape[:2]
        img_emb = torch.zeros(batch_size, n_regions, self.img_enc.out_dim).to(self.device)
        if torch.cuda.is_available():
            img_emb = img_emb.to(self.device)
    
        # 循环处理每个图像区域
        for i in range(n_regions):
            region_images = images[:, i, :, :, :]  # (batch_size, 3, 224, 224)
            img_emb[:, i, :] = self.img_enc(region_images)  # (batch_size, embed_dim)
        raw_img_emb = self.img_enc(raw_images)

        # EEG编码处理

        batch_size, n_band = eeg_data.shape[:2]
        eeg_emb = torch.zeros(batch_size, n_band, self.img_enc.out_dim).to(self.device)
        for i in range(n_band):
            band_eegs = eeg_data[:, i, :, :]
            if self.opt.eeg_encoder_type == "EEGTransformer":
                eeg_emb[:, i, :] = self.eeg_enc(band_eegs, self.chan_ids) # [batch_size, ]
                raw_eeg_emb = self.eeg_enc(raw_eeg_data, self.chan_ids) 
            elif self.opt.eeg_encoder_type == "TSLANet":
                eeg_emb[:, i, :] = self.eeg_enc(band_eegs)
                raw_eeg_emb = self.eeg_enc(raw_eeg_data)
            elif self.opt.eeg_encoder_type == "ATMS":
                subject_ids = torch.full((batch_size,), 1, dtype=torch.long).to(self.device)
                eeg_emb[:, i, :] = self.eeg_enc(band_eegs, subject_ids)
                raw_eeg_emb = self.eeg_enc(raw_eeg_data, subject_ids)
            elif self.opt.eeg_encoder_type == "BIOTEncoder":
                eeg_emb[:, i, :] = self.eeg_enc(band_eegs)
                raw_eeg_emb = self.eeg_enc(raw_eeg_data) 
            elif self.opt.eeg_encoder_type == "BENDR":
                subject_ids = torch.full((batch_size,), 1, dtype=torch.long).to(self.device)
                eeg_emb[:, i, :] = self.eeg_enc(band_eegs, subject_ids)
                raw_eeg_emb = self.eeg_enc(raw_eeg_data, subject_ids) 
            elif self.opt.eeg_encoder_type == "ACNet":
                eeg_emb[:, i, :] = self.eeg_enc(band_eegs)
                raw_eeg_emb = self.eeg_enc(raw_eeg_data) 
            elif self.opt.eeg_encoder_type == "FBCNet":
                eeg_emb[:, i, :] = self.eeg_enc(band_eegs)
                raw_eeg_emb = self.eeg_enc(raw_eeg_data) 
            elif self.opt.eeg_encoder_type == "EEGNet":
                eeg_emb[:, i, :] = self.eeg_enc(band_eegs.unsqueeze(1))  # [64, 17, 250] -> [64, 1, 17, 250]
                raw_eeg_emb = self.eeg_enc(raw_eeg_data.unsqueeze(1))  # [64, 17, 250] -> [64, 1, 17, 250]
            elif self.opt.eeg_encoder_type == "Conformer":
                eeg_emb[:, i, :] = self.eeg_enc(band_eegs.unsqueeze(1))  # [64, 17, 250] -> [64, 1, 17, 250]
                raw_eeg_emb = self.eeg_enc(raw_eeg_data.unsqueeze(1))  # [64, 17, 250] -> [64, 1, 17, 250]
            elif self.opt.eeg_encoder_type == "NICE":
                eeg_emb[:, i, :] = self.eeg_enc(band_eegs.unsqueeze(1))  # [64, 17, 250] -> [64, 1, 17, 250]
                raw_eeg_emb = self.eeg_enc(raw_eeg_data.unsqueeze(1))  # [64, 17, 250] -> [64, 1, 17, 250]
            elif self.opt.eeg_encoder_type == "ConvRNN":
                from types import SimpleNamespace
                subject_ids = torch.full((batch_size,), 1, dtype=torch.long).to(self.device)
                batch = SimpleNamespace(subject_index=subject_ids)
                inputs = {"meg": band_eegs}  # [batch_size, num_ch, time_steps]
                eeg_emb[:, i, :] = self.eeg_enc(inputs, batch)  # [batch_size, embed_size]
                inputs = {"meg": raw_eeg_data}  # [batch_size, num_ch, time_steps]
                raw_eeg_emb = self.eeg_enc(inputs, batch)  # [batch_size, embed_size]
            elif self.opt.eeg_encoder_type == "ETNet":
                eeg_emb[:, i, :] = self.eeg_enc(band_eegs)
                raw_eeg_emb = self.eeg_enc(raw_eeg_data)
 

        return img_emb, eeg_emb, raw_img_emb, raw_eeg_emb

    def forward_loss(self, img_emb, eeg_emb, raw_img_emb, raw_eeg_emb):
        """
        计算对比损失并添加全局损失
        img_emb:  (batch_size, n_regions, d)
        eeg_emb:  (batch_size,   n_bands,   d)
        raw_eeg_emb: (batch_size, d)
        raw_eeg_emb: (batch_size, d)
        """
        # 计算局部对比损失
        # local_loss = self.criterion(img_emb, eeg_emb)
        loss_high_beta, loss_high_gamma, loss_low_beta, loss_low_gamma = self.criterion(img_emb, eeg_emb)
        
        
        # 计算全局损失（使用MSE损失）
        global_loss = self.clip_loss_func(raw_img_emb, raw_eeg_emb, self.logit_scale)
        
        # 组合总损失
        # total_loss = self.lambda_local * local_loss + self.lambda_global * global_loss
        total_loss = self.lambda_high_beta * loss_high_beta + self.lambda_high_gamma * loss_high_gamma + \
                     self.lambda_low_gamma * loss_low_gamma + self.lambda_low_beta * loss_low_beta +\
                     self.lambda_global * global_loss
                
        
        # return local_loss.item() * self.lambda_local, global_loss.item() * self.lambda_global, total_loss
        return  loss_high_beta, loss_high_gamma, loss_low_beta, loss_low_gamma, global_loss, total_loss

    def train_emb(self, images, eeg_data, raw_images, raw_eeg_data):
        """训练一步"""
        self.Eiters += 1

        # 前向
        # img_emb shape: (batch_size, 4, 1024)
        # eeg_emb shape: (batch_size, 2, 1024)
        img_emb, eeg_emb, raw_image_emb, raw_eeg_emb = self.forward_emb(images, eeg_data, raw_images, raw_eeg_data)
        self.optimizer.zero_grad()
        # local_loss, global_loss, loss = self.forward_loss(img_emb, eeg_emb, raw_image_emb, raw_eeg_emb)
        loss_high_beta, loss_high_gamma, loss_low_beta, loss_low_gamma, global_loss, total_loss = self.forward_loss(img_emb, eeg_emb, raw_image_emb, raw_eeg_emb)

        # 反向+更新
        total_loss.backward()
        self.optimizer.step()

        # return local_loss, global_loss, loss 
        return loss_high_beta, loss_high_gamma, loss_low_beta, loss_low_gamma, global_loss, total_loss

