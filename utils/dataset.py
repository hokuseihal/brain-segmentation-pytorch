import torch
import numpy as np
from torchvision.transforms import ToTensor,Compose,Grayscale,ToPILImage,ColorJitter
import torchvision.transforms as T
from utils.augmentation import Crops,PositionJitter,Elastic_Distortion,Resize

import glob
import random
from PIL import Image
import os
import cv2
from utils.util import setcolor
def binary(x,a=.5):
    assert x.shape[0] == 3
    x[x>a]=1
    x[x<a]=0
    return x
class LinerCrackDataset(torch.utils.data.Dataset):
    def __init__(self,txt,size,**kwargs):
        self.size=size
        with open(txt) as f:
            self.txt=[l.strip().replace('jpg','txt') for l in f.readlines()]
        self.transform=T.Compose([T.Resize(size),T.ToTensor()])
    def __len__(self):
        return len(self.txt)
    def __getitem__(self,idx):
        im=Image.open(self.txt[idx].replace('txt','jpg'))
        mask=loadtxt(self.txt[idx])
        mask=np.floor(cv2.resize(mask,self.size)).astype(np.int64)
        #save mask
        maskimg=setcolor(torch.from_numpy(np.expand_dims(mask,axis=0)),torch.tensor([[0, 0, 0], [255, 255, 255], [0, 255, 0]]))[0]//255
        ToPILImage()(maskimg).save(self.txt[idx].replace('.txt','.jpg').replace('liner','liner/mask'))
        return self.transform(im),torch.from_numpy(mask)
def loadtxt(path):
    thickness=7
    def getdata(ind,sec='point'):
        for d in data:
            # print(d[0],ind)
            if len(d)==5 and d[0]==ind:
                if sec=='point':return int(d[1]),int(d[2])
                elif sec=='cls':return int(d[3])
        print(f"{path},{ind} is not found.")
    mask=np.zeros((800,800))
    with open(path) as f:
        data=[d.strip().split(',') for d in f.readlines()]
    # print(data)
    for d in data:
        try:
            if d[-1]=='0': continue
            if len(d)==5:
                cv2.line(mask,(int(d[1]),int(d[2])),getdata(d[-1]),color=int(d[3])+1,thickness=thickness)
            elif len(d)==2:
                cv2.line(mask,getdata(d[0]),getdata(d[1]),thickness=thickness,color=getdata(d[0],'cls')+1)
        except:
            print(f'ERROR on {path},{d}')
    return mask
class CrackDataset(torch.utils.data.Dataset):
    def __init__(self,rawp,maskp,transform=None,size=(256,256)):
        self.raw=sorted(glob.glob(f'{rawp}/*'))
        self.mask=[imgp for imgp in sorted(glob.glob(f'{maskp}/*')) if os.path.basename(imgp).split('.')[0] in [os.path.basename(rawimgp).split('.')[0] for rawimgp in self.raw]]
        self.in_channels=3
        self.out_channels=1
        self.transform=transform if transform is not None else Compose([Resize(size),ToTensor()])

        assert len(self.raw)==len(self.mask)
    def __len__(self):
        return len(self.raw)

    def __getitem__(self, item):
        imraw=Image.open(self.raw[item])
        immask=Image.open(self.mask[item])
        return self.transform(imraw),self.transform(immask)

class MulticlassCrackDataset(torch.utils.data.Dataset):
    def __init__(self,masks,transform=None,clscolor=[[0,0,0],[255,255,255],[0,255,0]],random=False,split=1,train=True,args=None,size=(256,256),half=False):
        assert split in {1,2,4,8,16}
        self.train=train
        self.mask=masks
        self.raw=[imgp.replace('mask','image') for imgp in self.mask]
        assert len(self.mask)==len(self.raw)
        self.in_channels=3
        self.out_channels=3
        self.size=size
        self.clscolor=torch.tensor(clscolor)//255
        self.random=random
        _transform=[]
        _posttransforms=[]
        _pretransforms=[]
        _pretransforms+=[Crops(self)]
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
        self.half=half
    def resize(self):
        print(self.size, '->', end='')
        # sz=64*random.randint(128//64,512//64)
        sz=64*random.randint(256//64,320//64)
        self.size=(sz, sz)
        print(self.size)
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
        mask=Image.open(self.mask[item]).convert('RGB')
        if self.half:
            if random.randint(0,1):
                region=(0,0,img.size[0]//2,img.size[1])
            else:
                region=(img.size[0] // 2,0, img.size[0],img.size[1])
            img=img.crop(region)
            mask=mask.crop(region)
        W,H=img.size
        if self.train:
            if self.random:
                self.cropshape=(random.randint(W//self.split,W),random.randint(H//self.split,H))
            else:
                self.cropshape=(W//self.split,H//self.split)
        sample=self.pretransforms({'image':img,'mask':mask,'posidx':posidx})
        img=self.transform(sample['image'])
        mask=binary(self.transform(sample['mask']))
        assert mask.shape[0] == 3
        sample=self.posttransforms({'image':img,'mask':mask})
        # ToPILImage()(sample['image']).show()
        # ToPILImage()(sample['mask']).show()
        # exit()
        img,mask=sample['image'],sample['mask']
        allmask=torch.zeros(self.size)
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
    linerdataset = LinerCrackDataset('../data/owncrack/liner', (4032,3024))
    for i in range(len(linerdataset)):
        print(i)
        linerdataset[i]