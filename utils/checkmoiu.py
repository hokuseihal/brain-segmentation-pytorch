import random
import glob
from PIL import Image
from torchvision.transforms import ToTensor
from util import miouf
import torch
def list_miou(outmask):
    white=torch.tensor([0.,0.,0.]).view(3,1)
    blue=torch.tensor([0.,0.,1.]).view(3,1)
    masks = glob.glob(f'/home/hokusei/src/data/owncrack/scene/mask/*.jpg')
    k_shot = int(len(masks) * 0.8)
    trainmask = random.sample(masks, k=k_shot)
    validmask = sorted(list(set(masks) - set(trainmask)))
    assert len(outmask[0])==len(validmask),f'{len(validmask)},{len(outmask[0])}'
    for i in range(len(validmask)):
        t=ToTensor()(Image.open(validmask[i]).resize((256,256)).convert('RGB'))
        t[:,t.sum(0)==3]=blue
        mious=[0 for _ in range(len(outmask))]
        for j in range(len(outmask)):
            out=ToTensor()(Image.open(outmask[j][i]).resize((256,256)).convert("RGB"))
            out[:,out.sum(0)==3]=blue
            mious[j]=miouf(out.unsqueeze(0),t.argmax(0),None)
        if (mious[3]>mious[2]) and (mious[3]>mious[1]) and (mious[2]>mious[0]) and (mious[1]>mious[0]):
            print(i,validmask[i])

if __name__=='__main__':
    mask1=glob.glob('/home/hokusei/src/crack-segmentation/data/out/normal1/*_out1.png')
    mask2 = glob.glob('/home/hokusei/src/crack-segmentation/data/out/focal2_1/*_out1.png')
    mask3 = glob.glob('/home/hokusei/src/crack-segmentation/data/out/split2_1/*_out.png')
    mask4 = glob.glob('/home/hokusei/src/crack-segmentation/data/out/split2focal_1/*_out.png')
    list_miou([mask1,mask2,mask3,mask4])
