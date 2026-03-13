import torch
import torch.nn as nn
from spikingjelly.clock_driven import layer
from mask_manage import PruningLayer
from snnvgg import myMultiStepIFNode


class SNNDVS5Conv(nn.Module):
    """
    论文 DVS-CIFAR10 backbone:
    64C3-AP2-128C3-AP2-128C3-AP2-256C3-AP2-256C3-AP2-10FC
    """

    def __init__(self, num_classes=10, in_channels=2):
        super().__init__()

        def conv_bn(out_c_in, out_c_out):
            return layer.SeqToANNContainer(
                nn.Conv2d(out_c_in, out_c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c_out),
            )

        def ap2():
            return layer.SeqToANNContainer(nn.AvgPool2d(2, 2))

        self.conv1 = conv_bn(in_channels, 64)
        self.sn1 = myMultiStepIFNode(detach_reset=True)
        self.pr1 = PruningLayer(layer_id=0)
        self.pool1 = ap2()

        self.conv2 = conv_bn(64, 128)
        self.sn2 = myMultiStepIFNode(detach_reset=True)
        self.pr2 = PruningLayer(layer_id=1)
        self.pool2 = ap2()

        self.conv3 = conv_bn(128, 128)
        self.sn3 = myMultiStepIFNode(detach_reset=True)
        self.pr3 = PruningLayer(layer_id=2)
        self.pool3 = ap2()

        self.conv4 = conv_bn(128, 256)
        self.sn4 = myMultiStepIFNode(detach_reset=True)
        self.pr4 = PruningLayer(layer_id=3)
        self.pool4 = ap2()

        self.conv5 = conv_bn(256, 256)
        self.sn5 = myMultiStepIFNode(detach_reset=True)
        self.pr5 = PruningLayer(layer_id=4)
        self.pool5 = ap2()
        self.gap = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))

        self.fc = nn.Linear(256, num_classes)

        # 给 manager 用（可选）
        self._sca_total_layers = 5

    def forward(self, x):
        """
        DVS batch 通常是 [B, T, C, H, W]
        你的 SeqToANNContainer 期望 [T, B, C, H, W]
        """
        if x.dim() == 5:
            # [B,T,C,H,W] -> [T,B,C,H,W]
            x = x.permute(1, 0, 2, 3, 4).contiguous()
        elif x.dim() == 4:
            # 如果你喂的是静态图，复用原逻辑：repeat 成 T=4（不建议用于 DVS）
            x = x.unsqueeze(0).repeat(10, 1, 1, 1, 1)
        else:
            raise ValueError(f'Unexpected input shape: {tuple(x.shape)}')

        out = self.conv1(x)
        out, v = self.sn1(out)
        out = self.pr1(out, v.detach())
        out = self.pool1(out)

        out = self.conv2(out)
        out, v = self.sn2(out)
        out = self.pr2(out, v.detach())
        out = self.pool2(out)

        out = self.conv3(out)
        out, v = self.sn3(out)
        out = self.pr3(out, v.detach())
        out = self.pool3(out)

        out = self.conv4(out)
        out, v = self.sn4(out)
        out = self.pr4(out, v.detach())
        out = self.pool4(out)

        out = self.conv5(out)
        out, v = self.sn5(out)
        out = self.pr5(out, v.detach())
        out = self.pool5(out)

        out = self.gap(out)

        # [T,B,256,1,1] -> [T,B,256]
        out = torch.flatten(out, 2)
        logits = self.fc(out.mean(dim=0))  # [B,10]
        return logits