import argparse
import os
import random
import glob
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

from core import save, addvalue, load, load_check, saveworter
from loss import DiceLoss, FocalLoss
from unet import UNet, wrapped_UNet
from utils.own import MulticlassCrackDataset as Dataset
from utils.util import miouf, prmaper
from gan import Discriminator
from radam import RAdam
from torch import autograd
from torch.autograd import Variable


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


def calculate_gradient_penalty(D,real_images, fake_images,lambda_term=10):
    B,C,H,W=real_images.shape
    device=real_images.device
    # eta = torch.FloatTensor(self.batch_size,1,1,1).uniform_(0,1)
    # eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
    # if self.cuda:
    #     eta = eta.cuda(self.cuda_index)
    # else:
    #     eta = eta
    eta=torch.rand(B,1,1,1).to(device)
    interpolated = eta * real_images + ((1 - eta) * fake_images)
    # define it to calculate gradient
    # interpolated = Variable(interpolated, requires_grad=True)
    # calculate probability of interpolated examples
    prob_interpolated = D(interpolated)
    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                              grad_outputs=torch.ones(
                                  prob_interpolated.size()).to(device),
                              create_graph=True, retain_graph=True)[0]
    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
    return grad_penalty





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
    discriminator=Discriminator(3).to(device)
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

    g_optimizer = RAdam(unet.parameters(), lr=args.lr)
    d_optimizer=RAdam(discriminator.parameters())

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
                g_optimizer.zero_grad()
                d_optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = unet(x)

                    gan_x=onehot(y_true)
                    d_fake_out=discriminator(y_pred).mean()
                    if i%args.num_d_train!=0:
                        print('\nd')
                        d_real_out=discriminator(gan_x).mean()

                        print(f'{epoch}:{i}/{len(loaders[phase])} EMD:{(d_real_out-d_fake_out).item():.4f}, d_real:{d_real_out.item():.4f}, d_fake:{d_fake_out.item():.4f}')
                        addvalue(writer,f'd_real:{phase}',d_real_out.item(),epoch)

                        if phase == "train":
                            gradient_penalty=calculate_gradient_penalty(discriminator,gan_x,y_pred)
                            addvalue(writer,f'EMD:{phase}',d_real_out-d_fake_out,epoch)
                            print(f' gp:{gradient_penalty.item():.4f}')
                            addvalue(writer, f'gp:{phase}', gradient_penalty, epoch)
                            (-d_real_out+d_fake_out+gradient_penalty).backward()
                            d_optimizer.step()
                            print('d_step')
                    else:
                        print('\ng')
                        print(f'{epoch}:{i}/{len(loaders[phase])} d_fake:{d_fake_out.item():.4f}')
                        if args.lambda_ce!=0:
                            celoss=F.cross_entropy(y_pred,y_true)
                            print(f'celoss:{celoss.item():.4f}')
                            if phase=='train':
                                (args.lambda_ce*celoss).backward(retain_graph=True)
                                addvalue(writer,f'celoss:{phase}',celoss.item(),epoch)
                        if phase == "train":
                            (-d_fake_out).backward()
                            g_optimizer.step()
                            print('g_step')
                    if phase=='train': addvalue(writer, f'd_fake:{phase}', d_fake_out.item(), epoch)
                    if phase == "valid":
                        miou = miouf(y_pred, y_true, len(traindataset.clscolor)).item()
                        valid_miou += [miou]
                        prmap += prmaper(y_pred, y_true, len(traindataset.clscolor))
                        if i == 0:
                            save_image(torch.cat(
                                [x, setcolor(y_true, traindataset.clscolor),
                                 setcolor(y_pred.argmax(1), traindataset.clscolor)],
                                dim=2), f'{args.savefolder}/{epoch}.jpg')
            print(f'{epoch=}/{args.epochs}:{phase}:{np.mean(losslist):.4f}')
            if phase == "valid":
                addvalue(writer, 'acc:miou', np.mean(valid_miou), epoch)
                print((prmap / (len(loaders[phase]) + 1)).int())
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
        default=1e-3,
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
    parser.add_argument(
        '--num_d_train',
        default=2,
        type=int
    )
    parser.add_argument(
        '--lambda_ce',
        default=0,
        type=float
    )
    args = parser.parse_args()
    args.num_train = args.split
    args.epochs *= args.split
    args.savefolder = f'data/{args.savefolder}'
    main(args)
