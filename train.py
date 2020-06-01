import argparse
import os
import random
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from core import save, addvalue
from loss import DiceLoss, FocalLoss
from utils.own import MulticlassCrackDataset as Dataset
from unet import UNet, wrapped_UNet
# from torch.utils.data import DataLoader
from utils.util import miouf,prmaper


def setcolor(idxtendor, colors):
    assert idxtendor.max() + 1 <= len(colors)
    B, H, W = idxtendor.shape
    colimg = torch.zeros(B, 3, H, W).to(idxtendor.device).to(idxtendor.device)
    colors = colors[1:]
    for b in range(B):
        for idx, color in enumerate(colors, 1):
            colimg[b, :, idxtendor[b] == idx] = (color.view(3, 1)).to(idxtendor.device).float()
    return colimg


import glob
def main(args):
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)
    print(device)

    masks=glob.glob(f'{args.maskfolder}/*.jpg')
    k_shot = int(len(masks) * 0.8) if args.k_shot == 0 else args.k_shot
    trainmask = random.sample(masks, k=k_shot)
    validmask = list(set(masks)-set(trainmask))
    traindataset = Dataset(trainmask,train=True,random=args.random,split=args.split)
    validdataset=Dataset(validmask,train=False,random=args.random,split=args.split)
    trainloader = torch.utils.data.DataLoader(traindataset, batch_size=args.batchsize, shuffle=True,
                                              num_workers=args.workers)
    validloader = torch.utils.data.DataLoader(validdataset, batch_size=args.batchsize, shuffle=True,
                                              num_workers=args.workers)
    loaders = {'train': trainloader, 'valid': validloader}
    unet = UNet(in_channels=traindataset.in_channels, out_channels=traindataset.out_channels, cutpath=args.cutpath)
    if args.pretrained:
        unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                              in_channels=3, out_channels=1, init_features=32, pretrained=True)
        unet = wrapped_UNet(unet, 1, traindataset.out_channels)
    if args.saveimg:unet.savefolder=args.savefolder
    unet.to(device)
    if args.loss == 'DSC':
        lossf = DiceLoss()
    elif args.loss == 'CE':
        lossf = nn.CrossEntropyLoss()
    elif args.loss == 'Focal':
        lossf = FocalLoss()
    else :
        assert False,'set correct loss.'

    optimizer = optim.Adam(unet.parameters(), lr=args.lr)
    writer = {}

    os.makedirs(args.savefolder, exist_ok=True)
    print('start train')
    for epoch in range(args.epochs):
        valid_miou = []
        for phase in ["train"] * args.num_train + ["valid"]:
            prmap=torch.zeros(len(traindataset.clscolor),len(traindataset.clscolor))
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

                    loss = lossf(y_pred, y_true)

                    addvalue(writer, f'loss:{phase}', loss.item(), epoch)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    if phase == "valid":
                        miou = miouf(y_pred, y_true, len(traindataset.clscolor)).item()
                        valid_miou+=[miou]
                        prmap+=prmaper(y_pred,y_true,len(traindataset.clscolor))
                        addvalue(writer, 'acc:miou', miou, epoch)
                        if i == 0: save_image(torch.cat(
                            [x, setcolor(y_true, traindataset.clscolor), setcolor(y_pred.argmax(1), traindataset.clscolor)],
                            dim=2), f'{args.savefolder}/{epoch}.jpg')
            print(f'{epoch=}/{args.epochs}:{phase}:{loss.item():.4f}')
            if phase == "valid":
                print(f'test:miou:{np.mean(valid_miou):.4f}')
                print((prmap/(i+1)).int())
        save(epoch, unet, args.savefolder, writer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training U-Net model for segmentation of brain MRI"
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=16,
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
    parser.add_argument(
        "--cutpath",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "--savefolder",
        default='data/tmp',
        type=str
    )
    parser.add_argument(
        "--rawfolder",
        default='../data/owncrack/scene/image',
        type=str
    )
    parser.add_argument(
        "--maskfolder",
        default='../data/owncrack/scene/mask',
        type=str
    )
    parser.add_argument(
        '--loss',
        default='CE',
    )
    parser.add_argument(
        '--split',
        type=int,
        default=1
    )
    parser.add_argument(
        '--random',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--saveimg',
        default=False,
        action='store_true'
    )
    args = parser.parse_args()
    args.num_train=args.split
    args.epochs*=args.split
    main(args)
