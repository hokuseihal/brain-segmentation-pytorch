import torch
import numpy as np
from torchvision.transforms import Resize,ToTensor,Compose,Grayscale,ToPILImage,ColorJitter
from utils.augmentation import Crops

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
    def __init__(self,masks,transform=None,clscolor=[[0,0,0],[255,255,255],[0,255,0]],random=False,split=1,train=True):
        assert split in {1,2,4,8,16}
        self.train=train
        self.mask=masks
        self.raw=[imgp.replace('mask','image') for imgp in self.mask]
        assert len(self.mask)==len(self.raw)
        self.in_channels=3
        self.out_channels=3
        self.clscolor=torch.tensor(clscolor)/255
        self.transform=transform if transform is not None else Compose([Resize((256,256)),ColorJitter(),ToTensor()])
        self.crops=Crops(self)
        self.random=random
        self.split=split
    def __len__(self):
        if self.train:
            return len(self.raw)
        else:
            return len(self.raw)*self.split**2
    def getposition(self,item):
        if self.train:
            return item,None
        else:
            posimg=item//(self.split**2)
            posw=(item%(self.split**2))//self.split
            posh=(item%self.split)
            # print('pos',item,posimg,posw,posh)
            return posimg,(posw,posh,self.split)
    def __getitem__(self, item):
        item,posidx=self.getposition(item)
        img=Image.open(self.raw[item])
        mask=Image.open(self.mask[item])
        W,H=img.size
        if self.train:
            if self.random:
                self.shape=(random.randint(W//self.split,W),random.randint(H//self.split,H))
            else:
                self.shape=(W//self.split,H//self.split)
        sample=self.crops({'image':img,'mask':mask,'posidx':posidx})
        # sample['image'].show()
        # sample['mask'].show()
        # exit()
        img=self.transform(sample['image'])
        immask=binary(self.transform(sample['mask']))

        allmask=torch.zeros_like(immask[0]).bool()
        clsmask=torch.zeros_like(allmask).long()
        for clsidx,color in enumerate(self.clscolor):
            color=color.view(3,1,1)
            clsmask[(immask==color).sum(0)==3]=clsidx
            allmask[(immask==color).sum(0)==3]=True

        assert (~allmask).float().mean()<1e-3
        assert clsmask.shape==(256,256)
        assert img.shape==(3,256,256)
        return img,clsmask
if __name__=='__main__':
    dataset=MulticlassCrackDataset('../data/owncrack/scene/image','../data/owncrack/scene/mask',random=False,train=False)
    for i in range(32):
        dataset[i]