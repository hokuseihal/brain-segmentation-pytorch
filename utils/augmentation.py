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
        return {'image':img,'mask':mask}
