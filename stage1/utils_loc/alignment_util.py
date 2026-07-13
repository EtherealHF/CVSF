import os
import sys
import time
import numpy as np
import torch
from collections import OrderedDict
from torch.autograd import Variable

from model.alignment.SCAN import xattn_score_e2i, xattn_score_i2e

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0): 
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        # for values that should be recorded exactly
        if self.count == 0:
            return str(self.val)
        return '%.4f (%.4f)' % (self.val, self.avg)

class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, repeat_time, log_step=10, logging=print):
    # 测试时, repeat_time == 1, EEG信号的数量和图片的数量就是一样的
    """Encode all images and EEG data loadable by `data_loader`."""
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # np array to keep all the embeddings
    img_embs = None
    eeg_embs = None

    for i, (image_features, eeg_data, image_features_raw, eeg_data_raw) in enumerate(data_loader):
        image_features = image_features.to("cuda:0")
        eeg_data = eeg_data.to("cuda:0")
        image_features_raw = image_features_raw.to("cuda:0")
        eeg_data_raw = eeg_data_raw.to("cuda:0")
        # image_features: [batch_size, 4, 3, 224, 224]
        # eeg_data: [batch_size, 2, C, T]
        # image_features_raw: [batch_size, 3, 224, 224]
        # eeg_data_raw: [batch_size, C, T] 
        model.logger = val_logger
        n_band = eeg_data.size(1)
        C = eeg_data.size(2)
        T = eeg_data.size(3)

        # compute the embeddings

        # =>
        # img_emb shape: [batch_size, 4, D]
        # eeg_emb shape: [batch_size, 2, D]
        # raw_img_emb shape: [batch_size, D]
        # raw_eeg_emb shape: [batch_size, D]
        img_emb, eeg_emb, raw_img_emb, raw_eeg_emb = model.forward_emb(image_features, eeg_data, image_features_raw, eeg_data_raw)
        # raw_img_emb_expanded = raw_img_emb.unsqueeze(1) 
        # raw_eeg_emb_expanded = raw_eeg_emb.unsqueeze(1)
        # img_emb = torch.cat([img_emb, raw_img_emb_expanded], dim=1)
        # eeg_emb = torch.cat([eeg_emb, raw_eeg_emb_expanded], dim=1)

        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
            eeg_embs = np.zeros((len(data_loader.dataset), eeg_emb.size(1), eeg_emb.size(2)))

            raw_img_embs = np.zeros((len(data_loader.dataset), raw_img_emb.size(1)))
            raw_eeg_embs = np.zeros((len(data_loader.dataset), raw_eeg_emb.size(1)))
        
        # Cache embeddings
        img_embs[i * data_loader.batch_size : (i + 1) * data_loader.batch_size] = img_emb.data.cpu().numpy().copy()
        eeg_embs[i * data_loader.batch_size : (i + 1) * data_loader.batch_size] = eeg_emb.data.cpu().numpy().copy()
        raw_img_embs[i * data_loader.batch_size : (i + 1) * data_loader.batch_size] = raw_img_emb.data.cpu().numpy().copy()
        raw_eeg_embs[i * data_loader.batch_size : (i + 1) * data_loader.batch_size] = raw_eeg_emb.data.cpu().numpy().copy()
        # measure accuracy and record loss
        # img_emb shape: (batch_size)
        model.forward_loss(img_emb, eeg_emb, raw_img_emb, raw_eeg_emb)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(i, len(data_loader), batch_time=batch_time,
                            e_log=str(model.logger)))
        del eeg_data, image_features
    # img_embs: [n_imgs, 4, D]
    # eeg_embs: [n_eegs, 2, D]
    return img_embs, eeg_embs, raw_img_embs, raw_eeg_embs



def shard_xattn_e2i(images, signals, opt, shard_size=1024):
    """Compute pairwise EEG->Image similarity with sharding"""
    n_im_shard = (len(images)-1)//shard_size + 1
    n_sig_shard = (len(signals)-1)//shard_size + 1
    d = np.zeros((len(images), len(signals)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        for j in range(n_sig_shard):
            sys.stdout.write('\r>> batch (%d,%d)' % (i, j))
            sig_start, sig_end = shard_size*j, min(shard_size*(j+1), len(signals))
            im = Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).cuda()
            s = Variable(torch.from_numpy(signals[sig_start:sig_end]), volatile=True).cuda()
            sim = xattn_score_e2i(im, s, opt)
            d[im_start:im_end, sig_start:sig_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d


def shard_xattn_i2e(images, signals, opt, shard_size=1024):
    """Compute pairwise Image->EEG similarity with sharding"""
    n_im_shard = (len(images)-1)//shard_size + 1
    n_sig_shard = (len(signals)-1)//shard_size + 1
    d = np.zeros((len(images), len(signals)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        for j in range(n_sig_shard):
            sys.stdout.write('\r>> batch (%d,%d)' % (i, j))
            sig_start, sig_end = shard_size*j, min(shard_size*(j+1), len(signals))
            im = Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).cuda()
            s = Variable(torch.from_numpy(signals[sig_start:sig_end]), volatile=True).cuda()
            sim = xattn_score_i2e(im, s, opt)
            d[im_start:im_end, sig_start:sig_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d

