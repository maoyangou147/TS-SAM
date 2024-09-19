import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict

class FeatureFusionDecoder(nn.Module):
    def __init__(self, c3_dims, factor=2):
        super(FeatureFusionDecoder, self).__init__()

        hidden_size = c3_dims // factor
        c2_size = c3_dims // factor
        c1_size = c3_dims // factor

        self.conv1 = nn.Sequential(
            nn.Conv2d(c3_dims, hidden_size, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(c2_size, hidden_size, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU()
        )
        maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        avgpool2 = nn.AvgPool2d(kernel_size=(2, 2))
        self.pool2 = nn.ModuleList([maxpool2, avgpool2])

        self.conv4 = nn.Sequential(
            nn.Conv2d(c1_size, hidden_size, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU()
        )
        maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))
        avgpool4 = nn.AvgPool2d(kernel_size=(2, 2))
        self.pool4 = nn.ModuleList([maxpool4, avgpool4])

        self.conv12_1 = nn.Sequential(
            nn.Conv2d(2 * hidden_size, hidden_size, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU()
        )
        self.conv12_2 = nn.Sequential(
            nn.Conv2d(2 * hidden_size, hidden_size, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU()
        )

        self.conv24_1 = nn.Sequential(
            nn.Conv2d(2 * hidden_size, hidden_size, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU()
        )
        self.conv24_2 = nn.Sequential(
            nn.Conv2d(2 * hidden_size, hidden_size, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU()
        )

        self.conv1_1 = nn.Conv2d(hidden_size, 1, 1)

    def forward(self, x_c3, x_c2, x_c1):
        # x_c3:1* x_c2:2* x_c1:4*
        x = self.conv1(x_c3)

        x_c2_down = self.conv2(x_c2)
        x_c2_down = self.pool2[0](x_c2_down) + self.pool2[1](x_c2_down)
        x_c1_down = self.conv4(x_c1)
        x_c1_down = self.pool4[0](x_c1_down) + self.pool4[1](x_c1_down)

        x = torch.cat([x, x_c2_down], dim=1)
        x = self.conv12_1(x)
        x = F.interpolate(input=x, size=(2 * x.size(-2), 2 * x.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c2], dim=1)
        x = self.conv12_2(x)

        x = torch.cat([x, x_c1_down], dim=1)
        x = self.conv24_1(x)
        x = F.interpolate(input=x, size=(2 * x.size(-2), 2 * x.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c1], dim=1)
        x = self.conv24_2(x)

        return self.conv1_1(x)