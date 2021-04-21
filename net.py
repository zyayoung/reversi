# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from layers import _Conv2d, _Linear, _BatchNorm2d
class ResBlock(nn.Module):
    def __init__(self, D_in, D, train):
        super().__init__()
        Conv2d = _Conv2d if train else nn.Conv2d
        BatchNorm2d = _BatchNorm2d if train else nn.BatchNorm2d
        self.h1 = Conv2d(D_in, D, 3, 1, 1, bias=False)
        self.bn1 = BatchNorm2d(D)
        self.relu1 = nn.ReLU()
        self.h2 = Conv2d(D, D, 3, 1, 1, bias=False)
        self.bn2 = BatchNorm2d(D)
        self.relu2 = nn.ReLU()
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        t = self.relu1(self.bn1(self.h1(x)))
        t = self.bn2(self.h2(t))
        return self.relu2(self.skip_add.add(x, t))


class ReversiNet(nn.Module):
    def __init__(self, device='cpu', cfg=[(8, 128)], train=False):
        super(ReversiNet, self).__init__()
        last_c = cfg[0][1]
        Conv2d = _Conv2d if train else nn.Conv2d
        Linear = _Linear if train else nn.Linear
        blocks = [Conv2d(3, last_c, 3), nn.ReLU()]
        for n, c in cfg:
            for i in range(n):
                blocks.append(ResBlock(last_c, c, train))
                last_c = c
        self.resblocks = nn.Sequential(*blocks)
        self.out = nn.Sequential(
            Conv2d(last_c, 64, 1),
            nn.ReLU(),
            nn.Flatten(),
            Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            Linear(512, 1024),
            nn.ReLU(),
        )

        self.out1 = Linear(1024, 1)
        self.out2 = Conv2d(last_c, 1, 1)
        self.device = device
        self.to(device)

        self.quant = torch.quantization.QuantStub()
        self.dequant1 = torch.quantization.DeQuantStub()
        self.dequant2 = torch.quantization.DeQuantStub()

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
        feat = self.out(x)
        p = self.dequant1(self.out1(feat))
        q = self.dequant2(self.out2(x)).flatten(1)
        return torch.cat([q, p], dim=-1)

    def convert(self, w: torch.tensor, b: torch.tensor):
        x = w.new_zeros((w.size(0), 2, 64))
        x_mask = torch.arange(0, 64).unsqueeze(0).to(x.device)
        x[:, 0] = (w.unsqueeze(1) >> x_mask) & 1
        x[:, 1] = (b.unsqueeze(1) >> x_mask) & 1
        return x.view(-1, 2, 8, 8).float()

    def pred(self, w, b):
        self(self.convert(w, b))
