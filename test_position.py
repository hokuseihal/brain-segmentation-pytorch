from utils.dataset import MulticlassCrackDataset as Dataset
import numpy as np
from core import load
import torch
from unet import UNet
from utils.util import prmaper,miouf
saved = load('data/split2')
writer, preepoch, modelpath, worter = saved['writer'], saved['epoch'], saved['modelpath'], saved['worter']
trainmask, validmask = worter['trainmask'], worter['validmask']
unet = UNet(in_channels=3, out_channels=3)
unet.load_state_dict(torch.load(modelpath))
print('load model')
device='cuda'
unet.to(device)
split=2
validdataset = Dataset(validmask, train=False, random=False, split=split)
validdataset.ret_item=True
prmap=np.zeros((split,split,3,3))
ioul=[[[] for _ in range(split)] for __ in range(split)]
meaniou=[]
unet.eval()
with torch.set_grad_enabled(False):
    for idx,(img,target,(item,pos)) in enumerate(validdataset):
        print(idx)
        img=img.to(device).unsqueeze(0)
        target=target.to(device).unsqueeze(0)
        output=unet(img)
        corx,cory,_=pos
        prmap[corx,cory]+=prmaper(output,target,3).detach().cpu().numpy()
        iou=miouf(output,target,3).item()
        ioul[corx][cory]+=[iou]
        meaniou+=[iou]
        print(np.nanmean(meaniou))
print(prmap)
for i in range(split):
    for j in range(split):
        print(np.nanmean(ioul[i][j]))