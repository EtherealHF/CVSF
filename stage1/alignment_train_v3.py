"""Training script"""
import os
import csv
import datetime
import time
import random
import shutil
import numpy as np
import yaml
import sys

import torch
STAGE1_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STAGE1_ROOT)
sys.path.insert(0, os.path.join(STAGE1_ROOT, "model", "EEG_Encoder", "braindecode_local"))
sys.path.insert(0, os.path.join(STAGE1_ROOT, "model", "EEG_Encoder", "brainmagick"))
sys.path.append(os.path.join(STAGE1_ROOT, "model", "EEG_Encoder", "brainmagic"))
from data.AlignmentEEGDataset import get_loaders
from model.alignment.SCAN import SCAN
from utils_loc.alignment_util import  AverageMeter, LogCollector, encode_data, shard_xattn_e2i, shard_xattn_i2e
from torch.autograd import Variable

import logging
import argparse
# import tensorboard_logger as tb_logger

# def set_seed(seed):
#     import torch
#     import numpy as np
#     import random
#     torch.manual_seed(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#     # cudnn设置
#     import torch.backends.cudnn as cudnn
#     cudnn.deterministic = True
#     cudnn.benchmark = False
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def main():
    parser = argparse.ArgumentParser()
    # 数据路径
    # EEG 参数
    parser.add_argument('--num_channels',    type=int, default=17)
    parser.add_argument('--sequence_length',type=int, default=250)
    parser.add_argument('--num_subjects',    type=int, default=2)
    parser.add_argument('--backbone', type=str, default="resnet34")
    # 训练超参
    parser.add_argument('--batch_size',      type=int, default=64)
    parser.add_argument('--num_epochs',      type=int, default=15)
    parser.add_argument('--learning_rate',   type=float, default=2e-4)
    parser.add_argument('--grad_clip',       type=float, default=2.0)
    parser.add_argument('--embed_size',      type=int, default=1024)
    parser.add_argument('--margin',          type=float, default=0.2)
    parser.add_argument('--max_violation',   action='store_true')
    parser.add_argument('--cross_attn',      choices=['img2eeg','eeg2img'], default='img2eeg')
    parser.add_argument('--raw_feature_norm', default="clipped_l2norm",
                        help='clipped_l2norm|l2norm|clipped_l1norm|l1norm|no_norm|softmax')
    parser.add_argument('--seed', type=int, default=38)
    # Logging & checkpoint
    parser.add_argument('--log_step',        type=int, default=10)
    parser.add_argument('--val_step',        type=int, default=500)
    parser.add_argument('--logger_name',     default='./runs/log')
    parser.add_argument('--model_name',      default='./outputs/SCAN/')
    parser.add_argument('--lambda_softmax', default=9., type=float,
                        help='Attention softmax temperature.')
    parser.add_argument('--lambda_lse', default=6., type=float,
                        help='LogSumExp temp.')
    parser.add_argument('--agg_func', default="LogSumExp",
                        help='LogSumExp|Mean|Max|Sum')
    parser.add_argument('--resume',          default='', type=str)
    parser.add_argument("--config", type=str, default="config/alignment.yaml", help="Path to config file")
    
    parser.add_argument('--lambda_global', type=float, default=1.0, help='Weight for global loss')
    parser.add_argument('--lambda_high_gamma', type=float, default=0.0, help='Weight for local loss')
    parser.add_argument('--lambda_high_beta', type=float, default=0.0, help='Weight for local loss')
    parser.add_argument('--lambda_low_gamma', type=float, default=0.0, help='Weight for local loss')
    parser.add_argument('--lambda_low_beta', type=float, default=0.0, help='Weight for local loss')

    parser.add_argument('--kway_eval', action='store_true', help='Use k-way (few-shot) evaluation')
    parser.add_argument('--k_list', type=str, default='4,200', help='k-way candidates, e.g. 4,200')
    parser.add_argument('--eeg_encoder_type', choices=['TSLANet', 'EEGTransformer', 'BIOTEncoder', 'ATMS', 'LaBraMEncoder', 'BENDR', 'ACNet', 'FBCNet', 'EEGNet', 'Conformer', 'NICE', 'ConvRNN', 'ETNet'], default='ATMS')
    parser.add_argument('--device', type=str, default='cuda:0', help='指定运行设备，例如cuda:0/cuda:1/cpu')
    parser.add_argument('--emb_dim', type=int, default=256, help='dimension of EEG encoder embedding (for TSLANet)')
    parser.add_argument('--sub_3000', type=str2bool, default=False) 
    

    opt = parser.parse_args()

    # set_seed(opt.seed)
    with open(opt.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config = config['datasets']
    config['sub_3000'] = opt.sub_3000
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')


    # 设置
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    # tb_logger.configure(opt.logger_name, flush_secs=5)
    torch.backends.cudnn.benchmark = True
    
    # 数据加载
    train_loader, val_loader = get_loaders(
        batch_size=opt.batch_size, workers=4, opt=opt, config=config)

    # 模型构建
    model = SCAN(opt)  # 用你写好的 EEG↔Image SCAN 类
    if opt.resume:
        ckpt = torch.load(opt.resume)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt['epoch'] + 1
    else:
        start_epoch = 0

    best_score = 0
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    for epoch in range(start_epoch, opt.num_epochs):
         adjust_learning_rate(model.optimizer, opt.learning_rate, epoch, opt)
         loss_high_beta, loss_high_gamma, loss_low_beta, loss_low_gamma, global_loss, total_loss = train(opt, train_loader, val_loader, model, epoch)
         logging.info(
             f"[Iter {model.Eiters}] "
             f"total_loss: {total_loss:.6f} | "
             f"global_loss: {global_loss:.6f} | "
             f"local_loss_sum: {(loss_high_beta + loss_high_gamma + loss_low_beta + loss_low_gamma):.6f} | "
             f"high_beta: {loss_high_beta:.6f} | "
             f"high_gamma: {loss_high_gamma:.6f} | "
             f"low_beta: {loss_low_beta:.6f} | "
             f"low_gamma: {loss_low_gamma:.6f}"
         )
         score = validate_wrapper(opt, val_loader, model, current_time, config)

         is_best = score > best_score
         best_score = max(score, best_score)
        #  save_checkpoint({
        #      'epoch': epoch,
        #      'model': model.state_dict(),
        #      'best_score': best_score,
        #  }, is_best, prefix=opt.model_name + '/')

         # 每5个epoch保存一次额外权重
         if (epoch + 1) % 1 == 0:
             lambda_dir = f"{opt.lambda_high_beta}_{opt.lambda_high_gamma}_{opt.lambda_low_beta}_{opt.lambda_low_gamma}_{opt.lambda_global}"
             epoch_ckpt = {
                 'epoch': epoch,
                 'model': model.state_dict(),
                 'best_score': best_score,
             }
             if config['sub_3000']:
                 subfix = "_3000"
             else:
                 subfix = "_16540"
             os.makedirs(os.path.join(opt.model_name, f"{lambda_dir}_{opt.eeg_encoder_type}_{opt.num_channels}_{subfix}"), exist_ok=True)
             torch.save(epoch_ckpt, os.path.join(opt.model_name, f"{lambda_dir}_{opt.eeg_encoder_type}_{opt.num_channels}_{subfix}" , f'scan_{epoch + 1}.pt'))


# -----------------------------------------
def train(opt, train_loader, val_loader, model, epoch):
    model.train_start()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    logger = LogCollector()
    end = time.time()
    total_loss_sum = 0
    loss_high_gamma_sum = 0
    loss_high_beta_sum = 0
    loss_low_gamma_sum = 0
    loss_low_beta_sum = 0
    global_loss_sum = 0

    # image: [64,2,3,224,224]
    # eeg_data: [64,2,17,250]
    # raw_images:[64,3,224,224]
    # raw_eeg_data: [64,17,250]
    for i, (images, eeg_data, raw_images, raw_eeg_data) in enumerate(train_loader):
        data_time.update(time.time() - end)
        model.logger = logger
        images = images.to(opt.device)
        eeg_data = eeg_data.to(opt.device)
        raw_images = raw_images.to(opt.device)
        raw_eeg_data = raw_eeg_data.to(opt.device)

        # local_loss, global_loss, loss = model.train_emb(images, eeg_data, raw_images, raw_eeg_data)
        loss_high_beta, loss_high_gamma, loss_low_beta, loss_low_gamma, global_loss, total_loss = model.train_emb(images, eeg_data, raw_images, raw_eeg_data)
        total_loss_sum += total_loss.item()
        # local_loss_sum += local_loss
        loss_high_beta_sum += loss_high_beta.item()
        loss_high_gamma_sum += loss_high_gamma.item()
        loss_low_beta_sum += loss_low_beta.item()
        loss_low_gamma_sum += loss_low_gamma.item()
        global_loss_sum += global_loss.item()

        batch_time.update(time.time() - end)
        end = time.time()

        if model.Eiters % opt.log_step == 0:
            logging.info(f'Epoch [{epoch}][{i}/{len(train_loader)}]\t'
                         f'{logger}\t'
                         f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         f'Data {data_time.val:.3f} ({data_time.avg:.3f})')
        # tb_logger.log_value('loss_high_beta', loss_high_beta_sum, step=model.Eiters)
        # tb_logger.log_value('loss_high_gamma', loss_high_gamma_sum, step=model.Eiters)
        # tb_logger.log_value('loss_low_beta', loss_low_beta_sum, step=model.Eiters)
        # tb_logger.log_value('loss_low_gamma', loss_low_gamma_sum, step=model.Eiters)
        # tb_logger.log_value('global_loss', global_loss_sum, step=model.Eiters)
        # tb_logger.log_value('total_loss', total_loss_sum, step=model.Eiters)


        # if model.Eiters % opt.val_step == 0:
        #     logging.info(f'[Iter {model.Eiters}] total_loss: {total_loss_sum:.6f} | local_loss: {local_loss_sum:.6f} | global_loss: {global_loss_sum:.6f}')
        #     validate(opt, val_loader, model)
    
    # return local_loss_sum, global_loss_sum, total_loss_sum
    return loss_high_beta_sum, loss_high_gamma_sum, loss_low_beta_sum, loss_low_gamma_sum, global_loss_sum, total_loss_sum

# -----------------------------------------
# def validate(opt, val_loader, model):
#     model.val_start()
#     # img_embs: [200, 4, D]
#     # eeg_embs: [200, 2, D]
#     # raw_img_embs: [200, D]
#     # raw_eeg_embs: [200, D]
#     img_embs, eeg_embs, raw_img_embs, raw_eeg_embs = encode_data(model, val_loader, repeat_time=4)
#     raw_img_embs_expanded = raw_img_embs.unsqueeze(1)  # [200, 1, D]
#     raw_eeg_embs_expanded = raw_eeg_embs.unsqueeze(1)  # [200, 1, D]

#     img_embs = torch.cat([img_embs, raw_img_embs_expanded], dim = 1) # [200, 5, D]
#     eeg_embs = torch.cat([eeg_embs, raw_eeg_embs_expanded], dim = 1) # [200, 3, D]
   
#     # shard 评估，避免一次性计算过大
#     if opt.cross_attn == 'img2eeg':
#         # 最终评估的是所训练的编码器的对齐效果, 因此这里直接将raw与分解后的数据拼接
#         # sims:  [200, 200]
#         sims = shard_xattn_i2e(img_embs, eeg_embs, opt, shard_size=1024)
#         # 评估 img->eeg 检索指标，例如 R@1, R@5
#         r1, r5, r10 = evaluate_i2e(sims)
#         logging.info(f'Image→EEG: R@1 {r1:.1f}, R@5 {r5:.1f}, R@10 {r10:.1f}')
#         score = r1 + r5 + r10
#     else:
#         sims = shard_xattn_e2i(eeg_embs, img_embs, opt, shard_size=1024)
#         r1, r5, r10 = evaluate_e2i(sims)
#         logging.info(f'EEG→Image: R@1 {r1:.1f}, R@5 {r5:.1f}, R@10 {r10:.1f}')
#         score = r1 + r5 + r10

#     tb_logger.log_value('val_score', score, step=model.Eiters)
#     return score
import torch
import os
import pandas as pd
import logging
from model.alignment.SCAN import xattn_score_i2e, xattn_score_e2i  # 确保导入

def validate(opt, val_loader, model, save_dir='./val_csv'):
    model.val_start()
    logging.info('validating...')
    
    # 提取所有嵌入
    img_embs, eeg_embs, raw_img_embs, raw_eeg_embs = encode_data(model, val_loader, repeat_time=4)
    
    # 扩展 raw_img_embs 和 raw_eeg_embs 以匹配维度
    raw_img_embs_expanded = np.expand_dims(raw_img_embs, axis=1)  # [N, 1, D]
    raw_eeg_embs_expanded = np.expand_dims(raw_eeg_embs, axis=1)  # [N, 1, D]
    img_embs = np.concatenate([img_embs, raw_img_embs_expanded], axis=1)  # [N, 5, D]
    eeg_embs = np.concatenate([eeg_embs, raw_eeg_embs_expanded], axis=1)  # [N, 3, D]

    print(f"n_images: {len(img_embs)}")
    print(f"n_eegs: {len(eeg_embs)}")
    # -------- Image→EEG --------
    sims_i2e = shard_xattn_i2e(img_embs, eeg_embs, opt, shard_size=1024)  # [N, N]
    r1_i2e, r5_i2e, r10_i2e, medr_i2e, meanr_i2e = evaluate_i2e(sims_i2e)
    logging.info(f'Image→EEG: R@1 {r1_i2e:.1f}, R@5 {r5_i2e:.1f}, R@10 {r10_i2e:.1f}')

    # -------- EEG→Image --------
    sims_e2i = shard_xattn_e2i(eeg_embs, img_embs, opt, shard_size=1024)  # [N, N]
    r1_e2i, r5_e2i, r10_e2i, medr_e2i, meanr_e2i = evaluate_e2i(sims_e2i)
    logging.info(f'EEG→Image: R@1 {r1_e2i:.1f}, R@5 {r5_e2i:.1f}, R@10 {r10_e2i:.1f}')

    # 记录到 Tensorboard
    # tb_logger.log_value('val_score_i2e', r1_i2e + r5_i2e + r10_i2e, step=model.Eiters)
    # tb_logger.log_value('val_score_e2i', r1_e2i + r5_e2i + r10_e2i, step=model.Eiters)

    # 保存指标到 CSV
    os.makedirs(save_dir, exist_ok=True)
    metrics = {
        'r1_i2e': [r1_i2e],
        'r5_i2e': [r5_i2e],
        'r10_i2e': [r10_i2e],
        'medr_i2e': [medr_i2e],
        'meanr_i2e': [meanr_i2e],
        'r1_e2i': [r1_e2i],
        'r5_e2i': [r5_e2i],
        'r10_e2i': [r10_e2i],
        'medr_e2i': [medr_e2i],
        'meanr_e2i': [meanr_e2i]
    }
    df = pd.DataFrame(metrics)
    metric_file = os.path.join(save_dir, "metrics.csv")
    df.to_csv(metric_file, mode='a', index=False, header=not os.path.exists(metric_file))

    # 返回所有指标
    return r1_e2i + r5_e2i + r10_e2i + r1_e2i +  r5_e2i + r10_e2i

def evaluate_i2e(sims):
    """
    Image→EEG (一对一)
    sims: (N, N) 相似度矩阵
    """
    npts = sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]  # 按相似度降序排列
        rank = np.where(inds == index)[0][0]  # 正确匹配EEG的排名
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * np.sum(ranks < 1) / len(ranks)
    r5 = 100.0 * np.sum(ranks < 5) / len(ranks)
    r10 = 100.0 * np.sum(ranks < 10) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    
    return r1, r5, r10, medr, meanr



def evaluate_e2i(sims):
    """
    EEG→Image (一对一)
    sims: (N, N) 相似度矩阵
    """
    npts = sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)

    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]  # 按相似度降序排列
        rank = np.where(inds == index)[0][0]  # 正确图片的排名
        ranks[index] = rank
        top1[index] = inds[0]

    # 计算指标
    print(f"npts: {npts}, len(ranks): {len(ranks)}")
    r1 = 100.0 * np.sum(ranks < 1) / len(ranks)
    r5 = 100.0 * np.sum(ranks < 5) / len(ranks)
    r10 = 100.0 * np.sum(ranks < 10) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return r1, r5, r10, medr, meanr


def extract_features_scan(scan, data_loader, device=None):
    scan.img_enc.eval()
    scan.eeg_enc.eval()
    image_feats = []
    eeg_feats = []
    with torch.no_grad():
        for batch in data_loader:
            _, _, images, eeg_datas = batch
            images = images.to(device)
            eeg_datas = eeg_datas.to(device)
            img_feat = scan.img_enc(images)
            if scan.opt.eeg_encoder_type == "EEGTransformer":
                eeg_feat = scan.eeg_enc(eeg_datas, scan.chan_ids)
            elif scan.opt.eeg_encoder_type == "TSLANet":
                eeg_feat = scan.eeg_enc(eeg_datas)
            elif scan.opt.eeg_encoder_type == "ATMS":
                batch_size = eeg_datas.size(0)
                subject_ids = torch.full((batch_size,), 1, dtype=torch.long).to(device)
                eeg_feat = scan.eeg_enc(eeg_datas, subject_ids)
            elif scan.opt.eeg_encoder_type == "BENDR":
                eeg_feat = scan.eeg_enc(eeg_datas)
            elif scan.opt.eeg_encoder_type == "BIOTEncoder":
                eeg_feat = scan.eeg_enc(eeg_datas)
            elif scan.opt.eeg_encoder_type == "ACNet":
                eeg_feat = scan.eeg_enc(eeg_datas)
            elif scan.opt.eeg_encoder_type == "FBCNet":
                eeg_feat = scan.eeg_enc(eeg_datas)
            elif scan.opt.eeg_encoder_type == "EEGNet":
                eeg_feat = scan.eeg_enc(eeg_datas.unsqueeze(1))  # 添加 unsqueeze(1)
            elif scan.opt.eeg_encoder_type == "Conformer":
                eeg_feat = scan.eeg_enc(eeg_datas.unsqueeze(1))  # 为 Conformer 添加支持
            elif scan.opt.eeg_encoder_type == "NICE":
                eeg_feat = scan.eeg_enc(eeg_datas.unsqueeze(1))  
            elif scan.opt.eeg_encoder_type == "ConvRNN":
                from types import SimpleNamespace
                batch_size = eeg_datas.size(0)
                subject_ids = torch.full((batch_size,), 1, dtype=torch.long).to(device)
                batch = SimpleNamespace(subject_index=subject_ids)
                inputs = {"meg": eeg_datas}  # 修改：将 Tensor 包装为字典
                eeg_feat = scan.eeg_enc(inputs, batch)  # 你模型应该支持 [B, 17, 250] 的输入
            elif scan.opt.eeg_encoder_type =="ETNet":
                eeg_feat = scan.eeg_enc(eeg_datas)

            image_feats.append(img_feat)
            eeg_feats.append(eeg_feat)
    all_image_feats = torch.cat(image_feats, dim=0)
    all_eeg_feats = torch.cat(eeg_feats, dim=0)
    return all_image_feats, all_eeg_feats

def k_way_accuracy(all_image_embeds, all_eeg_embeds, k_list=(4, 200)):
    n_samples = all_image_embeds.shape[0] # 所有输入的图片
    all_labels = set(range(n_samples))
    metrics = {}
    for k in k_list:
        image2eeg_correct = 0
        eeg2image_correct = 0
        correct_top5_i2e = 0
        correct_top5_e2i = 0
        # 用于收集命中样本的索引
        i2e_hit_indices = []
        i2e_top5_hit_indices = []
        e2i_hit_indices = []
        e2i_top5_hit_indices = []

        for current_class in range(n_samples):
            possible_classes = list(all_labels - {current_class})
            selected_classes = [current_class] + random.sample(possible_classes, k-1)
            current_image_feature = all_image_embeds[current_class]
            current_eeg_feature = all_eeg_embeds[current_class]
            selected_image_features = all_image_embeds[selected_classes]
            selected_eeg_features = all_eeg_embeds[selected_classes]
            logits_image2eeg = current_image_feature @ selected_eeg_features.T
            logits_eeg2image = current_eeg_feature @ selected_image_features.T

            _, top_idx_i2e = torch.topk(logits_image2eeg, 1) # 返回的是idx
            _, top_idx_e2i = torch.topk(logits_eeg2image, 1)
            # 记录 top1 命中
            if (top_idx_i2e == 0):
                image2eeg_correct += 1
                i2e_hit_indices.append(current_class)
            if (top_idx_e2i == 0):
                eeg2image_correct += 1
                e2i_hit_indices.append(current_class)
            # 记录 top5 命中
            if k >= 6: # k大于等于6才有top5
                _, top5_indices_i2e = torch.topk(logits_image2eeg, 5)
                _, top5_indices_e2i = torch.topk(logits_eeg2image, 5)
                if 0 in top5_indices_i2e.tolist():
                    correct_top5_i2e += 1
                    i2e_top5_hit_indices.append(current_class)
                if 0 in top5_indices_e2i.tolist():
                    correct_top5_e2i += 1
                    e2i_top5_hit_indices.append(current_class)

        print(f"\n=== {k}-way stats ===")
        print(f"image2eeg_{k}w_top1命中样本: {i2e_hit_indices} (共 {len(i2e_hit_indices)})")
        print(f"image2eeg_{k}w_top5命中样本: {i2e_top5_hit_indices} (共 {len(i2e_top5_hit_indices)})")
        print(f"eeg2image_{k}w_top1命中样本: {e2i_hit_indices} (共 {len(e2i_hit_indices)})")
        print(f"eeg2image_{k}w_top5命中样本: {e2i_top5_hit_indices} (共 {len(e2i_top5_hit_indices)})")

        metrics[f"image2eeg_{k}w_top1"] = image2eeg_correct / n_samples
        metrics[f"image2eeg_{k}w_top5"] = correct_top5_i2e / n_samples
        metrics[f"eeg2image_{k}w_top1"] = eeg2image_correct / n_samples
        metrics[f"eeg2image_{k}w_top5"] = correct_top5_e2i / n_samples 
    
    return metrics


import csv

def validate_kway(opt, val_loader, model, current_time, config, save_dir='./val_csv'):
    device = opt.device if torch.cuda.is_available() else "cpu"
    model.val_start()
    logging.info('validating (k-way)...')
    # 提特征
    img_feats, eeg_feats = extract_features_scan(model, val_loader, device=device)
    k_list = [int(k) for k in opt.k_list.split(',')]
    metrics = k_way_accuracy(img_feats, eeg_feats, k_list=k_list)
    logging.info(f'k-way metrics: {metrics}')

    os.makedirs(save_dir, exist_ok=True)
    # 推荐用参数写入文件名，防止覆盖
    
    if config['sub_3000']:
        subfix = "3000"
    else:
        subfix = "16540"
    
    if opt.num_channels == 17:
        subfix += "_17"
    else:
        subfix += "_63"
    metric_file = os.path.join(save_dir, current_time, f"kway_metrics_{opt.lambda_high_beta}_{opt.lambda_high_gamma}_{opt.lambda_low_beta}_{opt.lambda_low_gamma}_{opt.lambda_global}_{opt.eeg_encoder_type}_{subfix}.csv")

    os.makedirs(os.path.join(save_dir, current_time), exist_ok=True)
    
    # 用 dict writer 写入csv，不用 pandas
    file_exists = os.path.exists(metric_file)
    with open(metric_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)
        print(f'k-way metrics saved to {metric_file}')

    score = metrics[f'image2eeg_{k_list[0]}w_top1']
    return score

def validate_wrapper(opt, val_loader, model, current_time, config, save_dir='./val_csv'):
    if getattr(opt, 'kway_eval', False):
        return validate_kway(opt, val_loader, model, current_time, config, save_dir=save_dir)
    else:
        return validate(opt, val_loader, model, save_dir=save_dir)


# -----------------------------------------
def adjust_learning_rate(optimizer, base_lr, epoch, opt):
    lr = base_lr * (0.1 ** (epoch // 10))
    for g in optimizer.param_groups:
        g['lr'] = lr

# -----------------------------------------
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copy(prefix + filename, prefix + 'model_best.pth.tar')

# -----------------------------------------
if __name__ == '__main__':
    main()