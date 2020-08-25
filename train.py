import argparse
import torch.nn.functional as F
import os
import random
import glob
from net.sngan import SNResNetDiscriminator
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from core import save, addvalue, load, load_check, saveworter
from loss import DiceLoss, FocalLoss
from unet import UNet, wrapped_UNet
from utils.dataset import MulticlassCrackDataset as Dataset
from utils.util import miouf, prmaper
from gan import DC_Discriminator
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
    discriminator=SNResNetDiscriminator(in_ch=6).to(device)
    writer = {}
    worter = {}
    preepoch = 0

    def dis_hinge(dis_fake, dis_real):
        loss = torch.mean(torch.relu(1. - dis_real)) + \
               torch.mean(torch.relu(1. + dis_fake))
        return loss

    def gen_hinge(dis_fake, dis_real=None):
        return -torch.mean(dis_fake)
    if args.resume and load_check(args.savefolder):
        saved = load(args.savefolder)
        writer, preepoch, modelpath, worter = saved['writer'], saved['epoch'], saved['modelpath'], saved['worter']
        trainmask, validmask = worter['trainmask'], worter['validmask']
        unet.load_state_dict(torch.load(modelpath))
        print('load model')
    elif args.pretrained_G!='none' and load_check(args.pretrained_G):
        saved=load(args.pretrained_G)
        writer, _, modelpath, worter = saved['writer'], saved['epoch'], saved['modelpath'], saved['worter']
        trainmask, validmask = worter['trainmask'], worter['validmask']
        worter={}
        writer={}
        unet.load_state_dict(torch.load(modelpath))
        print('load model G')
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

    g_optimizer = RAdam(unet.parameters(), lr=1e-4)
    d_optimizer=RAdam(discriminator.parameters(),lr=1e-3)

    os.makedirs(args.savefolder, exist_ok=True)
    print('start train')
    for epoch in range(preepoch, args.epochs):
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
                B,C,H,W=x.shape
                g_optimizer.zero_grad()
                d_optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = unet(x)

                    gan_x=onehot(y_true)
                    if i%args.num_d_train!=0:
                        print('\nd')
                        d_fake_out=discriminator(torch.cat([x,y_pred.detach()],dim=1))
                        # fakeloss=F.binary_cross_entropy(d_fake_out,torch.zeros(B).to(device))
                        fakeloss=F.relu(1.+d_fake_out).mean()
                        addvalue(writer, f'd_fake:{phase}', fakeloss.item(), epoch)

                        d_real_out=discriminator(torch.cat([x,gan_x],dim=1))
                        # realloss=F.binary_cross_entropy(d_real_out,torch.ones(B).to(device))
                        realloss=F.relu(1.-d_real_out).mean()
                        print(f'{epoch}:{i}/{len(loaders[phase])}, d_real:{realloss.item():.4f}, d_fake:{fakeloss.item():.4f}')
                        addvalue(writer,f'd_real:{phase}',realloss.item(),epoch)
                        addvalue(writer, f'd_fake:{phase}', fakeloss.item(), epoch)
                        if phase == "train":
                            (realloss+fakeloss).backward()
                            d_optimizer.step()
                            print('d_step')
                    else:
                        print('\ng')
                        d_fake_out=discriminator(torch.cat([x,y_pred],dim=1))
                        # fakeloss=F.binary_cross_entropy(d_fake_out,torch.ones(B).to(device))
                        fakeloss=-(d_fake_out).mean()
                        addvalue(writer, f'g_fake:{phase}', fakeloss.item(), epoch)
                        print(f'{epoch}:{i}/{len(loaders[phase])} g_fake:{fakeloss.item():.4f}')
                        if args.lambda_ce!=0:
                            celoss=F.cross_entropy(y_pred,y_true)
                            print(f'celoss:{celoss.item():.4f}')
                            if phase=='train':
                                addvalue(writer,f'celoss:{phase}',celoss.item(),epoch)
                        if phase == "train":
                            (args.lambda_adv*fakeloss+args.lambda_ce*celoss).backward()
                            g_optimizer.step()
                            print('g_step')
                    if phase == "valid":
                        miou = miouf(y_pred, y_true, len(traindataset.clscolor)).item()
                        valid_miou += [miou]
                        prmap += prmaper(y_pred, y_true, len(traindataset.clscolor))
                        if i == 0:
                            save_image(torch.cat(
                                [x, setcolor(y_true, traindataset.clscolor),
                                 setcolor(y_pred.argmax(1), traindataset.clscolor)],
                                dim=2), f'{args.savefolder}/{epoch}.jpg')
            print(f'{epoch=}/{args.epochs}:{phase}')
            if phase == "valid":
                addvalue(writer, 'acc:miou', np.mean(valid_miou), epoch)
                print(f'miou:{np.mean(valid_miou):.4f}')
                print((prmap / (len(loaders[phase]) + 1)).int())
        save(epoch, unet, args.savefolder, writer, worter)
        torch.save(discriminator.state_dict(),f'{args.savefolder}/dis.pth')



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
    parser.add_argument(
        '--lambda_adv',
        default=1,
        type=float
    )
    parser.add_argument(
        "--pretrained_G",
        default='none',
    )
    args = parser.parse_args()
    args.num_train = args.split
    args.epochs *= args.split
    args.savefolder = f'data/{args.savefolder}'
    main(args)
