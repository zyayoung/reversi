# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from layers import _Conv2d, _Linear, _BatchNorm2d


class ReversiNet(nn.Module):
    def __init__(self, device='cpu', cfg=[(8, 128)], train=False):
        super(ReversiNet, self).__init__()
        last_c = cfg[0][1]
        Conv2d = _Conv2d if train else nn.Conv2d
        Linear = _Linear if train else nn.Linear
        self.resblocks = nn.Sequential(
            Conv2d(3, 32, 3, bias=False), _BatchNorm2d(32), nn.ReLU(),
            Conv2d(32, 32, 1, bias=False), _BatchNorm2d(32), nn.ReLU(),
            Conv2d(32, 64, 3, bias=False), _BatchNorm2d(64), nn.ReLU(),
            Conv2d(64, 64, 1, bias=False), _BatchNorm2d(64), nn.ReLU(),
            Conv2d(64, 128, 3, bias=False), _BatchNorm2d(128), nn.ReLU(),
            Conv2d(128, 128, 1, bias=False), _BatchNorm2d(128), nn.ReLU(),
            Conv2d(128, 256, 3, bias=False), _BatchNorm2d(256), nn.ReLU(),
            Conv2d(256, 256, 1, bias=False), _BatchNorm2d(256), nn.ReLU(),
            Conv2d(256, 512, 2, bias=False), _BatchNorm2d(512), nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Flatten(),
            Linear(512, 1024),
            nn.ReLU(),
            Linear(1024, 65),
        )
        self.device = device
        self.to(device)

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1))
        p = torch.zeros_like(x[:, 0, None])
        p[:, 0, 0] = 1
        p[:, 0, -1] = 1
        p[:, 0, :, 0] = 1
        p[:, 0, :, -1] = 1
        x = torch.cat([x, p], dim=1)

        x = self.quant(x)
        x = self.resblocks(x)
        return self.dequant(self.out(x))

    def convert(self, w: torch.tensor, b: torch.tensor):
        x = w.new_zeros((w.size(0), 2, 64))
        x_mask = torch.arange(0, 64).unsqueeze(0).to(x.device)
        x[:, 0] = (w.unsqueeze(1) >> x_mask) & 1
        x[:, 1] = (b.unsqueeze(1) >> x_mask) & 1
        return x.view(-1, 2, 8, 8).float()

    def pred(self, w, b):
        self(self.convert(w, b))
