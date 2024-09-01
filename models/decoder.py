import torch.nn as nn
import torch


def conv3x3(inplanes, outplanes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        inplanes,
        outplanes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(inplanes, outplanes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        shortcut=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride=1)
        self.upsample = None
        if stride != 1:
            self.upsample = nn.Upsample(scale_factor=stride, mode="bilinear")
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.shortcut = shortcut
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.upsample is not None:
            out = self.upsample(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        upsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.upsample layers upsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        inplanes,
        block,
        layers,
        groups=1,
        width_per_group=64,
        norm_layer=None,
    ):
        super(ResNet, self).__init__()
        self.inplanes = inplanes

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.dilation = 1
        layer_planes = [32, 64, 128, 256]
        layer_strides = [1, 2, 2, 1]

        self.groups = groups
        self.base_width = width_per_group

        self.layerf1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(200, 184, kernel_size=5, stride=1, padding=2, bias=False),
            norm_layer(184),
            nn.ReLU(inplace=True)
        )
        self.layer4 = self._make_layer(
            block, layer_planes[3], layers[3], stride=layer_strides[3]
        )
        self.layer3 = self._make_layer(
            block, layer_planes[2], layers[2], stride=layer_strides[2]
        )
        self.layer2 = self._make_layer(
            block, layer_planes[1], layers[1], stride=layer_strides[1]
        )
        self.layer1 = self._make_layer(
            block, layer_planes[1], layers[0], stride=layer_strides[0]
        )
        self.conv1 = nn.Conv2d(
            self.inplanes, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv2 = nn.Conv2d(self.inplanes, 3, kernel_size=1, stride=1, bias=False)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        shortcut = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            shortcut = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=1),
                nn.Upsample(scale_factor=stride, mode="bilinear"),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                shortcut,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    @property
    def layer0(self):
        return nn.Sequential(
            self.conv1, self.bn1, self.relu, self.upsample2, self.conv2
        )

    def forward(self, inputs):
        # f0 (B, 72, 32, 32)
        # f1 (B, 200, 16, 16)
        f0, f1 = inputs

        f1 = self.layerf1(f1)
        x = torch.cat([f0, f1], dim=1)
        for layer_idx in range(4, -1, -1):
            layer = getattr(self, f"layer{layer_idx}", None)
            if layer is not None:
                x = layer(x)
    
        return x


class ResUnet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        groups=1,
        width_per_group=64,
        norm_layer=None,
        inplanes=[64, 168, 256, 200],
        layer_planes=[32, 64, 128, 184]
    ):
        super(ResUnet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.dilation = 1

        layer_strides = [1, 2, 2, 2]

        self.groups = groups
        self.base_width = width_per_group

        self.layer4 = self._make_layer(
            block, inplanes[3], layer_planes[3], layers[3], stride=layer_strides[3]
        )
        self.layer3 = self._make_layer(
            block, inplanes[2], layer_planes[2], layers[2], stride=layer_strides[2]
        )
        self.layer2 = self._make_layer(
            block, inplanes[1], layer_planes[1], layers[1], stride=layer_strides[1]
        )
        self.layer1 = self._make_layer(
            block, inplanes[0], layer_planes[0], layers[0], stride=layer_strides[0]
        )
        inplanes = layer_planes[0] * block.expansion
        self.conv1 = nn.Conv2d(
            inplanes, inplanes, kernel_size=7, stride=1, padding=3, bias=False
        )
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv2 = nn.Conv2d(inplanes, 3, kernel_size=1, stride=1, bias=False)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        shortcut = None
        previous_dilation = self.dilation
        if stride != 1 or inplanes != planes * block.expansion:
            shortcut = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride=1),
                nn.Upsample(scale_factor=stride, mode="bilinear"),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                inplanes,
                planes,
                stride,
                shortcut,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    @property
    def layer0(self):
        return nn.Sequential(
            self.conv1, self.bn1, self.relu, self.upsample2, self.conv2
        )

    def forward(self, inputs):
        # f0 (B, 40, 64, 64)
        # f1 (B, 72, 32, 32)
        # f2 (B, 200, 16, 16)
        f0, f1, f2 = inputs

        o4 = self.layer4(f2)
        o3 = self.layer3(torch.cat([f1, o4], dim=1))
        o2 = self.layer2(torch.cat([f0, o3], dim=1))
        o1 = self.layer1(o2)

        out = self.layer0(o1)
        return out


def Decoder(feature_levels, block_type="basic"):
    if block_type == "basic":
        if feature_levels == 3:
            return ResUnet(block=BasicBlock, layers=[3, 4, 6, 3])
        else:
            return ResNet(inplanes=256, block=BasicBlock, layers=[3, 4, 6, 3])
    elif block_type == "bottle":
        if feature_levels == 3:
            return ResUnet(block=Bottleneck, layers=[3, 4, 6, 3])
        else:
            return ResNet(inplanes=256, block=Bottleneck, layers=[3, 4, 6, 3])
    else:
        raise NotImplementedError
