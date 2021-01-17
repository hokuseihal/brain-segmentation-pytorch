import torch
from unet import UNet
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image
import numpy as np
import cv2
def loadtxt(path,imgpath=None,colors=([0, 0, 0], [255,255,255], [0, 255, 0])):
    thickness=7
    def getdata(ind,sec='point'):
        for d in data:
            # print(d[0],ind)
            if len(d)==5 and d[0]==ind:
                if sec=='point':return int(d[1]),int(d[2])
                elif sec=='cls':return int(d[3])
        print(f"{path},{ind} is not found.")
    if imgpath is None:
        mask=np.zeros((800,800))
    else:
        mask=cv2.resize(cv2.imread(imgpath),(800,800))
    with open(path) as f:
        data=[d.strip().split(',') for d in f.readlines()]
    # print(data)
    for d in data:
        try:
            if d[-1]=='0': continue
            if len(d)==5:
                cv2.line(mask,(int(d[1]),int(d[2])),getdata(d[-1]),color=colors[int(d[3])+1],thickness=thickness)
            elif len(d)==2:
                cv2.line(mask,getdata(d[0]),getdata(d[1]),thickness=thickness,color=colors[getdata(d[0],'cls')+1])
        except:
            print(f'ERROR on {path},{d}')
    return mask
def setcolor(idxtendor, colors=torch.tensor([[0, 0, 0], [1.,1.,1.], [0, 1., 0]]),imgpath=None):
    assert idxtendor.max() + 1 <= len(colors)
    B, H, W = idxtendor.shape
    if imgpath is not None:
        colimg=T.ToTensor()(Image.open(imgpath).resize((size,size))).unsqueeze(0).to(idxtendor.device)
    else:
        colimg = torch.zeros(B, 3, H, W).to(idxtendor.device).to(idxtendor.device)
    colors = colors[1:]
    for b in range(B):
        for idx, color in enumerate(colors, 1):
            colimg[b, :, idxtendor[b] == idx] = (color.reshape(3, 1)).to(idxtendor.device).float()
    return colimg

def demo(imgpath='/home/hokusei/src/data/owncrack/img.jpg',modelpath='data/normal_x/model.pth',outpath='out.jpg',gt=None):
    img=T.ToTensor()(Image.open(imgpath).resize((size,size))).unsqueeze(0)
    model=UNet(out_channels=3)
    model.load_state_dict(torch.load(modelpath))
    model.eval()
    for p in model.parameters():
        p.requires_grad=False
    out=model(img)
    save_image(setcolor(out.argmax(1),imgpath=imgpath),outpath,imgpath)
    if gt is not None:
        cv2.imwrite(gt,loadtxt(imgpath.replace('jpg','txt'),imgpath))

if __name__=='__main__':
    import os
    imgfolder='datasets/rddliner'
    with open(f'{imgfolder}/val.txt') as f:
        imgp=[l.strip() for l in f.readlines()]
    size=640
    for p in imgp:
        if os.path.exists(p) and os.path.exists(p.replace('jpg','txt')):
            print(p)
            demo(p,'data/640_8/model.pth',f'out/{p.split("/")[-1]}',f'out_gt/{p.split("/")[-1]}')