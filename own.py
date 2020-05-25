import torch
from torchvision.transforms import Resize,ToTensor,Compose,Grayscale
import glob
import random
from PIL import Image
import os

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
    def __init__(self,rawp,maskp,transform=None,clscolor=[[0,0,0],[255,255,255],[0,0,255]]):
        self.mask=glob.glob(f'{maskp}/*')
        self.raw=[imgp for imgp in sorted(glob.glob(f'{rawp}/*')) if os.path.basename(imgp).split('.')[0] in [os.path.basename(rawimgp).split('.')[0] for rawimgp in self.mask]]
        assert len(self.mask)==len(self.raw)
        self.in_channels=3
        self.out_channels=1
        self.clscolor=torch.tensor(clscolor)/255
        #TODO CROP centor?
        self.transform=transform if transform is not None else Compose([Resize((256,256)),ToTensor()])

    def __len__(self):
        return len(self.raw)
    def __getitem__(self, item):
        imraw=self.transform(Image.open(self.raw[item]))
        immask=self.transform(Image.open(self.mask[item]))[:3]
        allmask=torch.zeros_like(immask[0]).bool()
        for clsidx,color in enumerate(self.clscolor):
            color=color.view(3,1,1)
            immask[(immask==color).sum(0)==3]=clsidx
            allmask[(immask==color).sum(0)==3]=True

        assert allmask.all()

        return imraw,immask.long()