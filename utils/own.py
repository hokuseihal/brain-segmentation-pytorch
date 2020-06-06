import torch
import numpy as np
from torchvision.transforms import Resize,ToTensor,Compose,Grayscale,ToPILImage,ColorJitter
from utils.augmentation import Crops,PositionJitter

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
        self.random=random
        self.pretransforms=Compose([Crops(self)])
        self.posttransforms=Compose([PositionJitter(3,1)])
        # self.posttransforms=Compose([])
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
        sample=self.pretransforms({'image':img,'mask':mask,'posidx':posidx})
        # sample['image'].show()
        # sample['mask'].show()
        # exit()
        img=self.transform(sample['image'])
        mask=binary(self.transform(sample['mask']))
        sample=self.posttransforms({'image':img,'mask':mask})
        img,mask=sample['image'],sample['mask']
        allmask=torch.zeros_like(mask[0]).bool()
        clsmask=torch.zeros_like(allmask).long()
        for clsidx,color in enumerate(self.clscolor):
            color=color.view(3,1,1)
            clsmask[(mask==color).sum(0)==3]=clsidx
            allmask[(mask==color).sum(0)==3]=True

        assert (~allmask).float().mean()<1e-3
        assert clsmask.shape==(256,256)
        assert img.shape==(3,256,256)
        return img,clsmask
import pickle
if __name__=='__main__':
    dataset=MulticlassCrackDataset(glob.glob('../data/owncrack/scene/mask/*'),random=False,train=True)
    _=dataset[0]
    with open('../tmpcrack/preout.pkl','rb') as f:
        preout=pickle.load(f)
    print((preout[1]!=_[1]).sum())