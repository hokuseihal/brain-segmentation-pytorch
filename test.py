from unet import UNet
import glob
from PIL import Image
import torch
from torchvision.transforms import Resize,ToTensor
import os
miouf=lambda pred,true,thresh=0.5:((pred>thresh)*(true>thresh)).sum().float()/((pred>thresh)+(true>thresh)).sum().float()
savedmodelp=f'{os.environ["HOME"]}/braindata/data/model.pth'
raws=f'{os.environ["HOME"]}/Pictures/paintset/raw/*.jpg'

iouthresh=0.7
model=UNet()
model.load_state_dict(torch.load(savedmodelp))
for imgp in glob.glob(raws):
    im=ToTensor()(Resize((256,256))(Image.open(imgp))).unsqueeze(0)
    target=ToTensor()(Resize((256,256))(Image.open(imgp.replace('raw','mask').replace('jpg','png'))))
    output=model(im)
    iou=miouf(output,target)
    if iou<iouthresh:
        print(imgp)
