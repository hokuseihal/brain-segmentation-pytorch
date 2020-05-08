import torch
from torchvision.transforms import Resize,ToTensor,Compose,Grayscale
import glob
from PIL import Image
class CrackDataset(torch.utils.data.Dataset):
    def __init__(self,raw,mask,transform=None):
        self.raw=sorted(glob.glob(raw))
        self.mask=sorted(glob.glob(mask))
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