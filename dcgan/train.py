from models_losses import *
from utils import *
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import argparse

parser = argparse.ArgumentParser(description='Argument parser')
parser.add_argument('--filter', type=int, default=128, help='Filtre sayisi')
parser.add_argument('--ogrenmehizi', type=float, default=0.0002, help='ogrenme hizini belirler, varsayilan deger 2 * 1e-4')
parser.add_argument('--cuda', type=int, default=0, help='1 ise gpu ustunde calisir')
parser.add_argument('--epoch', type=int, default=20, help='verisetin ustunden kac defa gecmek istediginiz, varsiyilan deger 20')
parser.add_argument('--batchsize', type=int, default=128, help='verisetinden bir ornekleme yapildiginda alinacak ornek sayisi')
opt = parser.parse_args()
d = opt.filter

img_size = 64
batch_size = opt.batchsize
dtype = torch.cuda.FloatTensor
lr = opt.ogrenmehizi

# Kullandıgım görselleştirme aracı
vis = visdom.Visdom(env='visual')

# Makalede anlatıldığı şekilde tüm verisetini 64*64'lük boyuta getirip normalize ediyoruz. 
transform = transforms.Compose([
        transforms.Scale(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Verisetini yükleyip üstünde gezilebilir(iteratable) bir hale getiriyoruz
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./datasets/MNIST', train=True, download=True,
                   transform=transform),
                   batch_size=batch_size, shuffle=True)

dtype = torch.FloatTensor
if opt.cuda == 1:
    dtype = torch.cuda.FloatTensor

lr = opt.ogrenmehizi
G = Generator(d)
D = Discriminator(d)

D.type(dtype)
G.type(dtype)

weight_init(G)
weight_init(D)


G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))


train(epochs, D, G, train_loader, D_optimizer, G_optimizer, batch_size, dtype)
