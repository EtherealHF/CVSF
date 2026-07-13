import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward

class DWTLoss(nn.Module):
    def __init__(self, wave='haar', mode='symmetric', level=1, device=None):
        """
        Args:
            wave: wavelet basis.
            mode: boundary handling mode.
            level: number of DWT levels.
        """
        super(DWTLoss, self).__init__()
        self.level = level
        self.dwt = DWTForward(J=level, wave=wave, mode=mode)

    def forward(self, pred, target):
        yl_pred, yh_pred = self.dwt(pred)
        yl_target, yh_target = self.dwt(target)
        loss_ll = F.mse_loss(yl_pred, yl_target, reduction='mean')

        loss_detail = 0.0
        for j in range(len(yh_pred)):
            for i in range(yh_pred[j].shape[2]):  # 閫氬父璇ョ淮搴﹀ぇ灏忎负 3
                loss_detail += F.mse_loss(yh_pred[j][:, :, i, :, :], yh_target[j][:, :, i, :, :], reduction='mean')

        total_loss = loss_ll + loss_detail
        return total_loss
