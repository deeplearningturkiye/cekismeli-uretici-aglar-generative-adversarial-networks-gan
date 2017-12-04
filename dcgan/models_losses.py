import torch
import torch.nn as nn
from torch.autograd import Variable


class Generator(nn.Module):

    def __init__(self, d=128):
        super(Generator, self).__init__()
        self.model1 = nn.Sequential(
            nn.ConvTranspose2d(100, d*8, 4, 1, 0),
            nn.BatchNorm2d(d*8),
            nn.ReLU())

        self.model2 = nn.Sequential(
            nn.ConvTranspose2d(d*8, d*4, 4, 2, 1),
            nn.BatchNorm2d(d*4),
            nn.ReLU())

        self.model3 = nn.Sequential(
            nn.ConvTranspose2d(d*4, d*2, 4, 2, 1),
            nn.BatchNorm2d(d*2),
            nn.ReLU())

        self.model4 = nn.Sequential(
            nn.ConvTranspose2d(d*2, d, 4, 2, 1),
            nn.BatchNorm2d(d),
            nn.ReLU())

        self.model5 = nn.Sequential(
            nn.ConvTranspose2d(d, 1, 4, 2, 1),
            nn.Tanh())

    def forward(self, x):
        out = self.model1(x)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)
        out = self.model5(out)
        return out


class Discriminator(nn.Module):

    def __init__(self, d=128):
        super(Discriminator, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(1, d, 4, 2, 1),
            nn.LeakyReLU(0.2))

        self.model2 = nn.Sequential(
            nn.Conv2d(d, d*2, 4, 2, 1),
            nn.BatchNorm2d(d*2),
            nn.LeakyReLU(0.2))

        self.model3 = nn.Sequential(
            nn.Conv2d(d*2, d*4, 4, 2, 1),
            nn.BatchNorm2d(d*4),
            nn.LeakyReLU(0.2))

        self.model4 = nn.Sequential(
            nn.Conv2d(d*4, d*8, 4, 2, 1),
            nn.BatchNorm2d(d*8),
            nn.LeakyReLU(0.2))

        self.model5 = nn.Sequential(
            nn.Conv2d(d*8, 1, 4, 1, 0),
            nn.Sigmoid())

    def forward(self, x):
        out = self.model1(x)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)
        out = self.model5(out)
        return out


def discriminator_loss(logits_real, logits_fake):
    
    bce_loss = nn.BCELoss()
    loss_real = bce_loss(logits_real, Variable(torch.ones(logits_real.size())).type(dtype))
    loss_fake = bce_loss(logits_fake, Variable(torch.zeros(logits_fake.size())).type(dtype))
    loss = (loss_real + loss_fake)
    return loss


def generator_loss(logits_fake):

    bce_loss = nn.BCELoss()
    loss = bce_loss(logits_fake, Variable(torch.ones(logits_fake.size())).type(dtype))
    return loss


def weight_init(model):
    for m in model.parameters():
            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or
                isinstance(m, nn.ConvTranspose2d) or
                isinstance(m, nn.BatchNorm2d)):
                nn.init.xavier_uniform(m[0])
                nn.init.xavier_uniform(m[1])
