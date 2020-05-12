import argparse
#import json
import os

import numpy as np
import torch
import torch.optim as optim
#from torch.utils.data import DataLoader
#from tqdm import tqdm

#from dataset import BrainSegmentationDataset as Dataset
#from logger import Logger
from loss import DiceLoss
#from transform import transforms
from unet import UNet
#from utils import log_images, dsc
from multiprocessing import cpu_count
from own import CrackDataset
from core import save,addvalue
from torchvision.utils import save_image
import random




def main(args):
    ##makedirs(args)
    ##snapshotargs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)
    print(device)
    #loader_train, loader_valid = data_loaders(args)
    #loaders = {"train": loader_train, "valid": loader_valid}
    dataset=CrackDataset('raw','mask')
    k_shot=int(len(dataset)*0.8) if args.k_shot==0 else args.k_shot
    trainidx=random.sample(range(len(dataset)),k=k_shot)
    validx=list(set(list(range(len(dataset))))-set(trainidx))
    traindataset=torch.utils.data.Subset(dataset,trainidx)
    validdataset=torch.utils.data.Subset(dataset,validx)
    #traindataset,validdataset=torch.utils.data.random_split(dataset,[n:=int(len(dataset)*0.8),len(dataset)-n])
    trainloader=torch.utils.data.DataLoader(traindataset,batch_size=args.batch_size,shuffle=True,num_workers=args.workers)
    validloader=torch.utils.data.DataLoader(validdataset,batch_size=args.batch_size,shuffle=True,num_workers=args.workers)
    loaders={'train':trainloader,'valid':validloader}
    unet = UNet(in_channels=dataset.in_channels, out_channels=dataset.out_channels)
    if args.pretrained:
        unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                               in_channels=3, out_channels=1, init_features=32, pretrained=True)
    unet.to(device)
    dsc_loss = DiceLoss()

    optimizer = optim.Adam(unet.parameters(), lr=args.lr)
    writer={}
    miouf=lambda pred,true,thresh=0.5:((pred>thresh)*(true>thresh)).sum().float()/((pred>thresh)+(true>thresh)).sum().float()
    os.makedirs('data',exist_ok=True)
    print('start train')
    for epoch in range(args.epochs):
        valid_miou=[]
        for phase in ["train"]*args.num_train+[ "valid"]:
            if phase == "train":
                unet.train()
            else:
                unet.eval()

            for i, data in enumerate(loaders[phase]):

                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = unet(x)

                    loss = dsc_loss(y_pred, y_true)

                    if phase == "train":
                        addvalue(writer,'loss:train',loss.item(),epoch)
                        loss.backward()
                        optimizer.step()
                    if phase == "valid":
                        addvalue(writer,'loss:valid',loss.item(),epoch)
                        miou=miouf(y_pred,y_true)
                        valid_miou.append(miou.item())
                        addvalue(writer,'acc:miou',miou,epoch)
                        if i==0:save_image(torch.cat([x,y_true.repeat(1,3,1,1),y_pred.repeat(1,3,1,1)],dim=2),f'data/{epoch}.jpg')
            print(f'{epoch=}/{args.epochs}:{phase}:{loss.item():.4f}')
            if phase=="valid":print(f'test:{np.mean(valid_miou)=:.4f}')
        save(epoch,unet,'data',writer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training U-Net model for segmentation of brain MRI"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="input batch size for training (default: 8)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for training (default: cuda:0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help="number of workers for data loading (default: max)",
    )
    parser.add_argument(
        "--vis-images",
        type=int,
        default=200,
        help="number of visualization images to save in log file (default: 200)",
    )
    parser.add_argument(
        "--vis-freq",
        type=int,
        default=10,
        help="frequency of saving images to log file (default: 10)",
    )
    parser.add_argument(
        "--weights", type=str, default="./weights", help="folder to save weights"
    )
    parser.add_argument(
        "--logs", type=str, default="./logs", help="folder to save logs"
    )
    parser.add_argument(
        "--images", type=str, default="./kaggle_3m", help="root folder with images"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="target input image size (default: 256)",
    )
    parser.add_argument(
        "--aug-scale",
        type=int,
        default=0.05,
        help="scale factor range for augmentation (default: 0.05)",
    )
    parser.add_argument(
        "--aug-angle",
        type=int,
        default=15,
        help="rotation angle range in degrees for augmentation (default: 15)",
    )
    parser.add_argument(
        "--pretrained",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "--k-shot",
        default=0,
        type=int
    )
    parser.add_argument(
        "--num-train",
        default=1,
        type=int
    )
    args = parser.parse_args()
    main(args)
