import torch
from torchvision.transforms import Resize,ToTensor,Compose,Grayscale,ToPILImage
import glob
import random
from PIL import Image
import os
def binary(x,a=.5):
    x[x>a]=1
    x[x<a]=0
    return x
class CrackDataset(torch.utils.data.Dataset):
    def __init__(self,rawp,maskp,transform=None):
        self.raw=sorted(glob.glob(f'{rawp}/*'))
        self.mask=[imgp for imgp in sorted(glob.glob(f'{maskp}/*')) if os.path.basename(imgp).split('.')[0] in [os.path.basename(rawimgp).split('.')[0] for rawimgp in self.raw]]
        self.in_channels=3
        self.out_channels=1
        self.transform=transform if transform is not None else Compose([Resize((256,256)),ToTensor()])

        assert len(self.raw)==len(self.mask)
    def __len__(self):
        return len(self.raw)

    def __getitem__(self, item):
        imraw=Image.open(self.raw[item])
        immask=Image.open(self.mask[item])
        return self.transform(imraw),self.transform(immask)
class MulticlassCrackDataset(torch.utils.data.Dataset):
    def __init__(self,rawp,maskp,transform=None,clscolor=[[0,0,0],[255,255,255],[0,255,0]]):
        self.mask=glob.glob(f'{maskp}/*')
        #self.raw=sorted([f'{rawp}/{raw_i}.jpg'  for raw_i in (set([i.split('.')[0] for i in  os.listdir(maskp)]) & set([i.split('.')[0] for i in  os.listdir(rawp)]))])
        self.raw=[imgp.replace('mask','image') for imgp in self.mask]
        assert len(self.mask)==len(self.raw)
        self.in_channels=3
        self.out_channels=3
        self.clscolor=torch.tensor(clscolor)/255
        #TODO CROP centor?
        self.transform=transform if transform is not None else Compose([Resize((256,256)),ToTensor()])

    def __len__(self):
        return len(self.raw)
    def __getitem__(self, item):
        imraw=self.transform(Image.open(self.raw[item]))
        immask=binary(self.transform(Image.open(self.mask[item]))[:3])

        allmask=torch.zeros_like(immask[0]).bool()
        clsmask=torch.zeros_like(allmask).long()
        for clsidx,color in enumerate(self.clscolor):
            color=color.view(3,1,1)
            clsmask[(immask==color).sum(0)==3]=clsidx
            allmask[(immask==color).sum(0)==3]=True
        
        assert (~allmask).float().mean()<1e-3
        assert clsmask.shape==(256,256)
        assert imraw.shape==(3,256,256)
        return imraw,clsmask