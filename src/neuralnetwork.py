import os
from typing import List, Tuple
import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None) -> None:
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def get_incoming_shape(incoming: torch.Tensor) -> List[int]:
    size = incoming.size()
    return [size[0], size[1], size[2], size[3]]

def interleave(tensors: List[torch.Tensor], axis: int) -> torch.Tensor:
    old_shape = get_incoming_shape(tensors[0])[1:]
    new_shape = [-1] + old_shape
    new_shape[axis] *= len(tensors)
    stacked = torch.stack(tensors, axis + 1)
    reshaped = stacked.view(new_shape)
    return reshaped

class UnpoolingAsConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UnpoolingAsConvolution, self).__init__()
        self.conv_A = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=1, padding=1)
        self.conv_B = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(2, 3), stride=1, padding=0)
        self.conv_C = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 2), stride=1, padding=0)
        self.conv_D = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(2, 2), stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_a = self.conv_A(x)
        padded_b = nn.functional.pad(x, (1, 1, 0, 1))
        output_b = self.conv_B(padded_b)
        padded_c = nn.functional.pad(x, (0, 1, 1, 1))
        output_c = self.conv_C(padded_c)
        padded_d = nn.functional.pad(x, (0, 1, 0, 1))
        output_d = self.conv_D(padded_d)
        left = interleave([output_a, output_b], axis=2)
        right = interleave([output_c, output_d], axis=2)
        y = interleave([left, right], axis=3)
        return y

class UpProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UpProjection, self).__init__()
        self.unpool_main = UnpoolingAsConvolution(in_channels, out_channels)
        self.unpool_res = UnpoolingAsConvolution(in_channels, out_channels)
        self.main_branch = nn.Sequential(
            self.unpool_main,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.residual_branch = nn.Sequential(
            self.unpool_res,
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        x = self.main_branch(input_data)
        res = self.residual_branch(input_data)
        x += res
        x = self.relu(x)
        return x

class ConConv(nn.Module):
    def __init__(self, in_channels_x1: int, in_channels_x2: int, out_channels: int) -> None:
        super(ConConv, self).__init__()
        self.conv = nn.Conv2d(in_channels_x1 + in_channels_x2, out_channels, kernel_size=1, bias=True)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = torch.cat([x2, x1], dim=1)
        x1 = self.conv(x1)
        return x1

class ResnetUnetHybrid(nn.Module):
    def __init__(self, block: nn.Module, layers: List[int], cfg_path: str = 'src/config.cfg') -> None:
        self.inplanes: int = 64
        super(ResnetUnetHybrid, self).__init__()
        self.config = self.parse_config(cfg_path)
        self.conv1 = nn.Conv2d(self.config['input_channels'], 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.conv2 = nn.Conv2d(2048, 1024, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(1024)
        self.up_proj1 = UpProjection(1024, 512)
        self.up_proj2 = UpProjection(512, 256)
        self.up_proj3 = UpProjection(256, 128)
        self.up_proj4 = UpProjection(128, 64)
        self.drop = nn.Dropout(0.5, False)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.con_conv1 = ConConv(1024, 512, 512)
        self.con_conv2 = ConConv(512, 256, 256)
        self.con_conv3 = ConConv(256, 128, 128)
        self.con_conv4 = ConConv(64, 64, 64)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def parse_config(self, cfg_path: str) -> dict:
        config = {}
        with open(cfg_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                key, value = line.strip().split('=')
                config[key.strip()] = int(value.strip())
        return config

    def _make_layer(self, block: nn.Module, planes: int, blocks: int, stride: int = 1) -> nn.Module:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x_to_conv4 = self.relu(x)
        x = self.maxpool(x_to_conv4)
        x_to_conv3 = self.layer1(x)
        x_to_conv2 = self.layer2(x_to_conv3)
        x_to_conv1 = self.layer3(x_to_conv2)
        x = self.layer4(x_to_conv1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.up_proj1(x)
        x = self.con_conv1(x, x_to_conv1)
        x = self.up_proj2(x)
        x = self.con_conv2(x, x_to_conv2)
        x = self.up_proj3(x)
        x = self.con_conv3(x, x_to_conv3)
        x = self.up_proj4(x)
        x = self.con_conv4(x, x_to_conv4)
        x = self.drop(x)
        x = self.conv3(x)
        x = self.relu(x)
        return x

    @classmethod
    def load_pretrained(cls, device: torch.device, load_path: str = 'model/hyb_net_weights.model') -> 'ResnetUnetHybrid':
        model = cls(Bottleneck, [3, 4, 6, 3])
        if not os.path.exists(load_path):
            print('Downloading model weights...')
            os.system('wget https://www.dropbox.com/s/amad4ko9opi4kts/hyb_net_weights.model')
        model = model.to(device)
        model.load_state_dict(torch.load(load_path, map_location=device))
        return model
