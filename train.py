import argparse
import os
import random
import glob
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from core import save, addvalue, load, load_check, saveworter
from loss import DiceLoss, FocalLoss
from unet import UNet, wrapped_UNet
from utils.own import MulticlassCrackDataset as Dataset
from utils.util import miouf, prmaper


def setcolor(idxtendor, colors):
    assert idxtendor.max() + 1 <= len(colors)
    B, H, W = idxtendor.shape
    colimg = torch.zeros(B, 3, H, W).to(idxtendor.device).to(idxtendor.device)
    colors = colors[1:]
    for b in range(B):
        for idx, color in enumerate(colors, 1):
            colimg[b, :, idxtendor[b] == idx] = (color.view(3, 1)).to(idxtendor.device).float()
    return colimg
def onehot(x,num_class=3):
    return torch.eye(num_class)[x].permute(0,3,1,2).to(x.device)



def main(args):
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)
    print(device)

    masks = glob.glob(f'{args.maskfolder}/*.jpg')
    k_shot = int(len(masks) * 0.8) if args.k_shot == 0 else args.k_shot
    trainmask = random.sample(masks, k=k_shot)
    validmask = list(set(masks) - set(trainmask))

    unet = UNet(in_channels=3, out_channels=3, cutpath=args.cutpath)
    if args.pretrained:
        unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                              in_channels=3, out_channels=1, init_features=32, pretrained=True)
        unet = wrapped_UNet(unet, 1, 3)
    writer = {}
    worter = {}
    preepoch = 0
    if args.resume and load_check(args.savefolder):
        saved = load(args.savefolder)
        writer, preepoch, modelpath, worter = saved['writer'], saved['epoch'], saved['modelpath'], saved['worter']
        trainmask, validmask = worter['trainmask'], worter['validmask']
        unet.load_state_dict(torch.load(modelpath))
        print('load model')
    unet.to(device)
    saveworter(worter, 'trainmask', trainmask)
    saveworter(worter, 'validmask', validmask)
    traindataset = Dataset(trainmask, train=True, random=args.random, split=args.split)
    validdataset = Dataset(validmask, train=False, random=args.random, split=args.split)
    trainloader = torch.utils.data.DataLoader(traindataset, batch_size=args.batchsize, shuffle=True,
                                              num_workers=args.workers)
    validloader = torch.utils.data.DataLoader(validdataset, batch_size=args.batchsize, shuffle=True,
                                              num_workers=args.workers)
    loaders = {'train': trainloader, 'valid': validloader}
    if args.saveimg: unet.savefolder = args.savefolder
    if args.loss == 'DSC':
        lossf = DiceLoss()
    elif args.loss == 'CE':
        lossf = nn.CrossEntropyLoss()
    elif args.loss == 'Focal':
        lossf = FocalLoss()
    elif args.loss=='L1':
        lossf=lambda output,target:F.l1_loss(output,onehot(target))
    else:
        assert False, 'set correct loss.'

    optimizer = optim.Adam(unet.parameters(), lr=args.lr)

    os.makedirs(args.savefolder, exist_ok=True)
    print('start train')
    for epoch in range(preepoch, args.epochs):
        losslist = []
        valid_miou = []
        for phase in ["train"] * args.num_train + ["valid"]:
            prmap = torch.zeros(len(traindataset.clscolor), len(traindataset.clscolor))
            if phase == "train":
                unet.train()
                if args.resize:
                    traindataset.resize()
            else:
                unet.eval()

            for i, data in enumerate(loaders[phase]):
                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = unet(x)

                    loss = lossf(y_pred, y_true)
                    losslist += [loss.item()]
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    if phase == "valid":
                        miou = miouf(y_pred, y_true, len(traindataset.clscolor)).item()
                        valid_miou += [miou]
                        prmap += prmaper(y_pred, y_true, len(traindataset.clscolor))
                        if i == 0:
                            save_image(torch.cat(
                                [x, setcolor(y_true, traindataset.clscolor),
                                 setcolor(y_pred.argmax(1), traindataset.clscolor)],
                                dim=2), f'{args.savefolder}/{epoch}.jpg')
            addvalue(writer, f'loss:{phase}', np.mean(losslist), epoch)
            print(f'{epoch=}/{args.epochs}:{phase}:{np.mean(losslist):.4f}')
            if phase == "valid":
                print(f'test:miou:{np.mean(valid_miou):.4f}')
                addvalue(writer, 'acc:miou', np.mean(valid_miou), epoch)
                print((prmap / (i + 1)).int())
        save(epoch, unet, args.savefolder, writer, worter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training U-Net model for segmentation of brain MRI"
    )
    parser.add_argument(
        "--batchsize",
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
        default="cuda",
        help="device for training (default: cuda)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help="number of workers for data loading (default: max)",
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
        default='tmp',
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
    parser.add_argument(
        '--resume',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--resize',
        default=False,
        action='store_true'
    )
    args = parser.parse_args()
    args.num_train = args.split
    args.epochs *= args.split
    args.savefolder = f'data/{args.savefolder}'
    main(args)
