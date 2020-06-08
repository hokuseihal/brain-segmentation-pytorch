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
            assert len(self.dataset.cropshape)==2
            assert type(self.dataset.cropshape[0])==int
            assert type(self.dataset.cropshape[1])==int
            #random crop
            x0=random.randint(0,W-self.dataset.cropshape[0])
            y0=random.randint(0,H-self.dataset.cropshape[1])
            width,height=self.dataset.cropshape
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
        Cimg,H,W=img.shape
        Cmask,H,W=mask.shape
        flag=False
        if random.randint(0,1)==0:
            img=img.permute(0,2,1)
            mask=mask.permute(0,2,1)
            flag=True
        for i in range(0,H,self.block):
            _jit=random.randint(0,self.jit)
            if _jit!=0:
                if random.randint(0,1)==0:
                    img[:,i]=torch.cat([img[:,i,_jit:],torch.zeros(Cimg,_jit)],dim=-1)
                    mask[:,i]=torch.cat([mask[:,i,_jit:],torch.zeros(Cmask,_jit)],dim=-1)
                else:
                    img[:,i]=torch.cat([torch.zeros(Cimg,_jit),img[:,i,:-_jit]],dim=-1)
                    mask[:,i]=torch.cat([torch.zeros(Cmask,_jit),mask[:,i,:-_jit]],dim=-1)
        if flag:
            img=img.permute(0,2,1)
            mask=mask.permute(0,2,1)
        samples['image']=img
        samples['mask']=mask
        # from torchvision.transforms import ToPILImage
        # ToPILImage()(img).show()
        # exit()
        return samples
import numpy
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from torchvision.transforms import ToPILImage
import torchvision.transforms.functional as TF
class Elastic_Distortion(object):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    def __init__(self,sigma=6,alpha=None,random_state=None):
        self.alpha=alpha
        self.sigma=sigma
        self.random_state=random_state
    def __call__(self,sample):
        if self.random_state is None:
            random_state = numpy.random.RandomState(None)
        if self.alpha is None:
            alpha=random.randint(0,50)
        sample['image']=sample['image'].numpy()
        sample['mask']=sample['mask'].numpy()
        image=sample['image']
        mask=sample['mask']
        C,H,W=image.shape
        shape = (H,W)
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * alpha

        x, y = numpy.meshgrid(numpy.arange(shape[0]), numpy.arange(shape[1]),indexing='ij')
        for c in range(C):
            indices = numpy.reshape(x+dy, (-1, 1)), numpy.reshape(y+dx, (-1, 1))
            sample['image'][c]=map_coordinates(image[c], indices, order=1).reshape(shape)
            sample['mask'][c]=map_coordinates(mask[c], indices, order=1).reshape(shape)
        sample['image']=torch.from_numpy(sample['image'])
        sample['mask']=torch.from_numpy(sample['mask'])
        # ToPILImage()(sample['image']).show()
        # ToPILImage()(sample['mask']).show()
        # exit()
        return sample
class Resize(object):
    def __init__(self,dataset):
        self.dataset=dataset
    def __call__(self, img):
        img=img.resize(self.dataset.shape)
        return img