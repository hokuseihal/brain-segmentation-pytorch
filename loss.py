import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc
class FocalLoss(nn.Module):
    def __init__(self,gamma=2):
        super(FocalLoss,self).__init__()
        self.gamma=gamma
    def foward(self,y_pred,y_true):
        #y_pred(B,C,H,W)
        #y_true(B,H,W)
        y_pred=F.softmax(y_pred,1)
        pt=torch.gather(y_pred,1,y_true)
        return (-((1-pt)**self.gamma)*torch.log(pt)).mean()