import torch
from multiprocessing import cpu_count
from unet import UNet
import os
from radam import RAdam
from core import addvalue,savedic
from utils.util import miouf
import glob
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
#dataset
class BDD_road_Seg_Dataset(torch.utils.data.Dataset):
    def __init__(self,root,seg='train'):
        self.root=root
        self.images=sorted(glob.glob(f'{root}/images/{seg}/*'))
        self.masks=sorted(glob.glob(f'{root}/labels/{seg}/*'))
        assert len(self.images) == len(self.masks)
        self.size=(256,256)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        im=ToTensor()(Image.open(self.images[idx]).resize(self.size))
        mask=(ToTensor()(Image.open(self.masks[idx]).resize(self.size))==0).float().squeeze(0)
        return im,mask
def operate(phase):
    if phase=='train':
        unet.train()
        loader=trainloader
    else:
        unet.eval()
        loader=validloader

    with torch.set_grad_enabled(phase=='train'):
        for bchidx,(data,target) in enumerate(loader):
            data=data.to(device)
            target=target.to(device)
            output=unet(data)
            loss=criterion(output,target)
            print(f'{loss.item():.4f}')
            if phase=='train':
                (loss/subdivisions).backward()
            if (bchidx*(batchsize//subdivisions))%(batchsize)==0:
                print(f'{e}:{phase}{bchidx}/{len(loader)}:step')
                optimizer.step()
                optimizer.zero_grad()
            addvalue(writer,f'loss:{phase}',loss.item(),e)
            miou=miouf(output,target,2)
            addvalue(writer,f'miou:{phase}',miou.item(),e)
            if bchidx==0:
                save_image(torch.cat([data,output.repeat(1,3,1,1)],dim=-1),f'{folder}/{e}.png')
if __name__=='__main__':
    writer={}
    device = torch.device("cpu" if not torch.cuda.is_available() else 'cuda')
    print(device)
    unet=UNet(3,1).to(device)
    optimizer = RAdam(unet.parameters())
    criterion=torch.nn.BCELoss()
    folder='data/roadseg'
    os.makedirs('data/roadseg/', exist_ok=True)
    batchsize=64
    subdivisions=2
    num_cpu=cpu_count()
    trainloader=torch.utils.data.DataLoader(BDD_road_Seg_Dataset('../data/bdd100k/seg'),batch_size=batchsize//subdivisions,shuffle=True,num_workers=num_cpu)
    validloader=torch.utils.data.DataLoader(BDD_road_Seg_Dataset('../data/bdd100k/seg',seg='val'),batch_size=batchsize//subdivisions,shuffle=True,num_workers=num_cpu)
    for e in range(100):
        operate('train')
        operate('valid')
        savedic(writer,f'data/{folder}')
        torch.save(unet.state_dict(),'data/roadseg/model.pth')
