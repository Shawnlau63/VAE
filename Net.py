import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  # N,64,14,14
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )  # N,128,7,7
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )  # N,256,3,3
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )  # N,512,1,1
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )  # N,256,1,1
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 2, 1, 1),
            # nn.BatchNorm2d(2),
            # nn.ReLU(inplace=True)
        )  # N,2,1,1

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        y4 = self.conv4(y3)
        y5 = self.conv5(y4)
        y6 = self.conv6(y5)
        logsigma = y6[:, :1, :, :]
        miu = y6[:, 1:, :, :]

        return logsigma, miu


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )  # N,256,1,1
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )  # N,512,1,1
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )  # N,256,3,3
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )  # N,128,7,7
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  # N,64,14,14
        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 3, 2, 1, output_padding=1),
            nn.Sigmoid()
        )  # N,1,28,28

    def forward(self, x, logsigma, miu):
        x = x * torch.exp(logsigma) + miu
        x = x.permute([0, 3, 1, 2])
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        y4 = self.conv4(y3)
        y5 = self.conv5(y4)
        y6 = self.conv6(y5)

        return y6