import glob
import torch
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from PIL import Image
from unet import UNet
from utils.util import miouf,prmaper
valids=''
savedmodelpath=''
saveimgpath=''
model=UNet()
thresh=0.5
model=model.load_state_dict(torch.load(savedmodelpath))
for imgp in valids:
    x=ToTensor()(Image.open(imgp))
    t=ToTensor()(Image.open(imgp.replace('image','mask')))
    y=model(x)
    miou=miouf(y,t,None)
    prmap=prmaper(y,t,3)
    if miou<thresh:
        print(imgp,miou,prmap)
        save_image(torch.cat([x,t,y],dim=1),f'{saveimgpath}/{imgp.split("/")[-1]}')
