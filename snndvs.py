
from mask_manage import PruningLayer
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from spikingjelly.clock_driven import functional, surrogate, layer, neuron
from typing import Callable, overload

import torch
import torch.nn as nn
from spikingjelly.clock_driven import layer, neuron

from mask_manage import PruningLayer
from snnvgg import myMultiStepIFNode  # 复用你已有的 IF 多步节点


class SNNDVS5Conv(nn.Module):
    """
    Paper-style: 64C3-AP2-128C3-AP2-128C3-AP2-256C3-AP2-256C3-AP2-10FC
    Compatible with your SCA pruning pipeline (PruningLayer + myMultiStepIFNode).
    """

    def __init__(self, num_classes=10, in_channels=2, T=10):
        super().__init__()
        self.T = int(T)

        # ---- block 1: in -> 64 ----
        self.layer1 = layer.SeqToANNContainer(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.neuron1 = myMultiStepIFNode(detach_reset=True)
        self.prune1 = PruningLayer(layer_id=0,)
        self.pool1 = layer.SeqToANNContainer(nn.AvgPool2d(kernel_size=2, stride=2))

        # ---- block 2: 64 -> 128 ----
        self.layer2 = layer.SeqToANNContainer(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.neuron2 = myMultiStepIFNode(detach_reset=True)
        self.prune2 = PruningLayer(layer_id=1)
        self.pool2 = layer.SeqToANNContainer(nn.AvgPool2d(kernel_size=2, stride=2))

        # ---- block 3: 128 -> 128 ----
        self.layer3 = layer.SeqToANNContainer(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.neuron3 = myMultiStepIFNode(detach_reset=True)
        self.prune3 = PruningLayer(layer_id=2)
        self.pool3 = layer.SeqToANNContainer(nn.AvgPool2d(kernel_size=2, stride=2))

        # ---- block 4: 128 -> 256 ----
        self.layer4 = layer.SeqToANNContainer(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.neuron4 = myMultiStepIFNode(detach_reset=True)
        self.prune4 = PruningLayer(layer_id=3)
        self.pool4 = layer.SeqToANNContainer(nn.AvgPool2d(kernel_size=2, stride=2))

        # ---- block 5: 256 -> 256 ----
        self.layer5 = layer.SeqToANNContainer(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.neuron5 = myMultiStepIFNode(detach_reset=True)
        self.prune5 = PruningLayer(layer_id=4)
        self.pool5 = layer.SeqToANNContainer(nn.AvgPool2d(kernel_size=2, stride=2))

        # classifier (time-major)
        self.fc = nn.Linear(256, num_classes)

        # 用于让 manager 动态设置 total_layers（也支持你不改 manager 时手动设）
        self._sca_total_layers = 5

    def forward(self, x):
        """
        x can be:
          - [B, C, H, W]         (static) -> repeat to [T,B,C,H,W]
          - [T, B, C, H, W]      (time-major)
          - [B, T, C, H, W]      (batch-major) -> permute to time-major
        """
        if x.dim() == 4:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        elif x.dim() == 5:
            # try to normalize to [T,B,C,H,W]
            if x.shape[0] == self.T:
                pass
            elif x.shape[1] == self.T:
                x = x.permute(1, 0, 2, 3, 4).contiguous()
            else:
                # 允许数据集给的 T != self.T：按输入的 T 走
                # 默认认为第一维是 T（更常见）
                pass
        else:
            raise ValueError(f'Unexpected input shape: {tuple(x.shape)}')

        out = self.layer1(x)
        out, v1 = self.neuron1(out)
        out = self.prune1(out, v1.detach())
        out = self.pool1(out)

        out = self.layer2(out)
        out, v2 = self.neuron2(out)
        out = self.prune2(out, v2.detach())
        out = self.pool2(out)

        out = self.layer3(out)
        out, v3 = self.neuron3(out)
        out = self.prune3(out, v3.detach())
        out = self.pool3(out)

        out = self.layer4(out)
        out, v4 = self.neuron4(out)
        out = self.prune4(out, v4.detach())
        out = self.pool4(out)

        out = self.layer5(out)
        out, v5 = self.neuron5(out)
        out = self.prune5(out, v5.detach())
        out = self.pool5(out)

        # [T,B,256,1,1] -> flatten spatial -> [T,B,256]
        out = torch.flatten(out, 2)
        logits = self.fc(out.mean(dim=0))  # [B,num_classes]
        return logits