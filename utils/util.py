import torch
import torchvision
import numpy as np
def miouf(_pred,_t_idx,_):
    pred=_pred.clone()
    t_idx=_t_idx.clone()
    B,numcls,H,W=pred.shape
    with torch.no_grad():
        pred=pred.argmax(1)
        miou=[]
        for clsidx in range(1,numcls):
            if not (t_idx==clsidx).any():
                continue
            iou=(((pred==clsidx) & (t_idx==clsidx)).sum())/(((pred==clsidx) | (t_idx==clsidx)).sum().float())
            miou+=[iou.item()]
        miou=np.mean(miou)
        return miou

def prmaper(_pred,_t_idx,numcls):
    pred=_pred.clone()
    t_idx=_t_idx.clone()
    with torch.no_grad():
        pred=pred.argmax(1)
        prmap=torch.zeros(numcls,numcls)
        for pred_i in range(numcls):
            for t_i in range(numcls):
                prmap[pred_i,t_i]=((pred==pred_i) & (t_i==t_idx)).sum()
        return prmap
