import torchvision.transforms.functional as TF
import random
class Crops(object):
    def __init__(self,dataset):
        self.dataset=dataset
    def __call__(self, samples):
        img,mask,posidx=samples['image'],samples['mask'],samples['posidx']
        W,H=img.size
        # print(f'size:{img.size}')

        if self.dataset.train:
            assert len(self.dataset.shape)==2
            assert type(self.dataset.shape[0])==int
            assert type(self.dataset.shape[1])==int
            #random crop
            x0=random.randint(0,W-self.dataset.shape[0])
            y0=random.randint(0,H-self.dataset.shape[1])
            width,height=self.dataset.shape
        else:
            #postion crop for valid
            posw,posh,s=posidx
            x0=W//s*posw
            y0=H//s*posh
            width=W//s
            height=H//s
        # print(posidx)
        # print(img.size)
        # print(x0,y0,width,height)
        img=TF.crop(img,y0,x0,height,width)
        # print(x0,y0,width,height)
        mask=TF.crop(mask,y0,x0,height,width)
        # img.show()
        samples['image']=img
        samples['mask']=mask
        return samples
import torch
class PositionJitter(object):
    def __init__(self,jit,block):
        self.jit=jit
        self.block=block
    def __call__(self, samples):
        img=samples['image']
        mask=samples['mask']
        C,H,W=img.shape
        if random.randint(0,1)==0 or True:
            img=img.permute(0,2,1)
            mask=mask.permute(0,2,1)
        for i in range(0,H,self.block):
            _jit=random.randint(0,self.jit)
            if _jit!=0:
                if random.randint(0,1)==0:
                    img[:,i]=torch.cat([img[:,i,:-_jit],torch.zeros(C,_jit)],dim=-1)
                    mask[:,i]=torch.cat([mask[:,i,:-_jit],torch.zeros(C,_jit)],dim=-1)
                else:
                    img[:,i]=torch.cat([torch.zeros(C,_jit),img[:,i,_jit:]],dim=-1)
                    mask[:,i]=torch.cat([torch.zeros(C,_jit),mask[:,i,_jit:]],dim=-1)
        img=img.permute(0,2,1)
        mask=mask.permute(0,2,1)
        samples['image']=img
        samples['mask']=mask
        from torchvision.transforms import ToPILImage
        ToPILImage()(img).show()
        ToPILImage()(mask).show()
        exit()
        return mask
