"""The fixed two-stage U-Net shown in the feature explorer."""

import torch
from torch import nn
from torch.nn import functional as F


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc_l1_conv1 = nn.Conv2d(1, out_channels=16, kernel_size=3, padding=1)
        self.enc_l1_conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_l2_conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.enc_l2_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck_conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bottleneck_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        # Decoder
        self.dec_l2_deconv = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.dec_l2_conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.dec_l2_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.dec_l1_deconv = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.dec_l1_conv1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.dec_l1_conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)

        self.final_conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)

    def forward(self, x):
        enc1 = F.relu(self.enc_l1_conv1(x))
        enc1 = F.relu(self.enc_l1_conv2(enc1))
        enc2 = F.relu(self.enc_l2_conv1(self.pool1(enc1)))
        enc2 = F.relu(self.enc_l2_conv2(enc2))
        features = F.relu(self.bottleneck_conv1(self.pool2(enc2)))
        features = F.relu(self.bottleneck_conv2(features))
        features = torch.cat([self.dec_l2_deconv(features), enc2], dim=1)
        features = F.relu(self.dec_l2_conv1(features))
        features = F.relu(self.dec_l2_conv2(features))
        features = torch.cat([self.dec_l1_deconv(features), enc1], dim=1)
        features = F.relu(self.dec_l1_conv1(features))
        features = F.relu(self.dec_l1_conv2(features))
        return self.final_conv(features)
