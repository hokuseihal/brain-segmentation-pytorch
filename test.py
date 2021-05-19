import numpy as np
from unet import UNet
import glob
from PIL import Image
import torch
from torchvision.transforms import Resize,ToTensor
from torchvision.utils import save_image
import os
import torch.nn.functional as F
miouf=lambda pred,true,thresh=0.5:((pred>thresh)*(true>thresh)).sum().float()/((pred>thresh)+(true>thresh)).sum().float()
savedmodelp=f'{os.environ["HOME"]}/braindata/normal/model.pth'
raws=f'../data/owncrack/above/image/*.jpg'
savedcutmodelp=savedmodelp.replace('normal','cutpath')
iouthresh=0.7
model=UNet()
model.load_state_dict(torch.load(savedmodelp))
cutmodel=UNet(cutpath=True,out_channels=3)
cutmodel.load_state_dict(torch.load(savedcutmodelp))

msel=[]

for idx,imgp in enumerate(glob.glob(raws)):
    im=ToTensor()(Resize((256,256))(Image.open(imgp))).unsqueeze(0)
    target=ToTensor()(Resize((256,256))(Image.open(imgp.replace('raw','mask'))))
    output=model(im)
    cutoutput=cutmodel(im)[:,1]
    if (mse:=F.mse_loss(cutoutput,output).item())>0.02:
        output=output.repeat(1,3,1,1)
        cutoutput=cutoutput.repeat(1,3,1,1)
        target=torch.cat([target.unsqueeze(0),output,cutoutput],dim=2)
        save_image(target,f'data/comp/above/{idx}.jpg')
    print(idx,mse)
