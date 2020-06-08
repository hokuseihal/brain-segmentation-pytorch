import torch
import numpy as np
from torchvision.transforms import ToTensor,Compose,Grayscale,ToPILImage,ColorJitter
from utils.augmentation import Crops,PositionJitter,Elastic_Distortion,Resize

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
    def __init__(self,masks,transform=None,clscolor=[[0,0,0],[255,255,255],[0,255,0]],random=False,split=1,train=True,args=None):
        assert split in {1,2,4,8,16}
        self.train=train
        self.mask=masks
        self.raw=[imgp.replace('mask','image') for imgp in self.mask]
        assert len(self.mask)==len(self.raw)
        self.in_channels=3
        self.out_channels=3
        self.shape=(256,256)
        self.clscolor=torch.tensor(clscolor)/255
        self.random=random
        _transform=[]
        _posttransforms=[]
        _pretransforms=[]
        _pretransforms+=[Crops(self)]
        if train :
            _transform+=[ColorJitter()]
            if args.jitter >0:
                _posttransforms+=[PositionJitter(args.jitter,args.jitter_block)]
            if args.elastic:
                _posttransforms+=[Elastic_Distortion()]
        _transform+=[Resize(self),ToTensor()]
        self.pretransforms=Compose(_pretransforms)
        self.transform=Compose(_transform)
        self.posttransforms=Compose(_posttransforms)
        self.split=split
        self.ret_item=False
        print(f'{self.train=}')
        print(f'{self.pretransforms=}')
        print(f'{self.transform=}')
        print(f'{self.posttransforms=}')
    def resize(self):
        print(self.shape,'->',end='')
        sz=64*random.randint(128//64,512//64)
        self.shape=(sz,sz)
        print(self.shape)
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
            return posimg,(posw,posh,self.split)
    def __getitem__(self, item):
        item,posidx=self.getposition(item)
        img=Image.open(self.raw[item])
        mask=Image.open(self.mask[item])
        W,H=img.size
        if self.train:
            if self.random:
                self.cropshape=(random.randint(W//self.split,W),random.randint(H//self.split,H))
            else:
                self.cropshape=(W//self.split,H//self.split)
        sample=self.pretransforms({'image':img,'mask':mask,'posidx':posidx})
        # sample['image'].show()
        # sample['mask'].show()
        # exit()
        img=self.transform(sample['image'])
        mask=binary(self.transform(sample['mask']))
        sample=self.posttransforms({'image':img,'mask':mask})
        img,mask=sample['image'],sample['mask']
        allmask=torch.zeros(self.shape)
        clsmask=torch.zeros_like(allmask).long()
        # from torchvision.transforms import ToPILImage
        # ToPILImage()(img).show()
        # ToPILImage()(mask).show()
        for clsidx,color in enumerate(self.clscolor):
            color=color.view(3,1,1)
            clsmask[(mask==color).sum(0)==3]=clsidx
            allmask[(mask==color).sum(0)==3]=1
        if self.ret_item:
            return img,clsmask,(item,posidx)
        assert allmask.bool().any()
        return img,clsmask
import pickle
if __name__=='__main__':
    dataset=MulticlassCrackDataset(glob.glob('../data/owncrack/scene/mask/*'),random=False,train=True)
    _=dataset[0]
    with open('../tmpcrack/preout.pkl','rb') as f:
        preout=pickle.load(f)
    print((preout[1]!=_[1]).sum())