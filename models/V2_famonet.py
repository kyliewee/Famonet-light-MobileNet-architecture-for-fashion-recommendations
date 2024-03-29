#import torch
import torch.nn as nn
from torchsummary import summary

class SeparableConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):

        super().__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(
            input_channels,
            input_channels,
            kernel_size,
            groups = input_channels,
            **kwargs
            )
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(
            input_channels, 
            output_channels, 
            kernel_size = 1)
        )

        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):

        super().__init__()

        self.conv = nn.Conv2d(
            input_channels, 
            output_channels, 
            kernel_size, 
            **kwargs
        )
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class MobileNet(nn.Module):

    def __init__(self, class_num = 10, width_multiplier = 1):
       super().__init__()

       alpha = width_multiplier
       # set the input channel = 1 for grayscale image
       self.stem = nn.Sequential(
           BasicConv2d(
           1, 
           int(32 * alpha), 
           kernel_size = 3, 
           stride = 1, 
           padding = 1, 
           bias = False
           ),
           SeparableConv2d(
           int(32 * alpha),
           int(64 * alpha),
           kernel_size = 3,
           padding = 1,
           bias = False
           )
       )

       #downsample
       self.conv1 = nn.Sequential(
           SeparableConv2d(
               int(64 * alpha),
               int(128 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),
           SeparableConv2d(
               int(128 * alpha),
               int(128 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv2 = nn.Sequential(
           SeparableConv2d(
               int(128 * alpha),
               int(256 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),
           SeparableConv2d(
               int(256 * alpha),
               int(256 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv3 = nn.Sequential(
           SeparableConv2d(
               int(256 * alpha),
               int(512 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),

           SeparableConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           SeparableConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           SeparableConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           SeparableConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           SeparableConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv4 = nn.Sequential(
           SeparableConv2d(
               int(512 * alpha),
               int(1024 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),
           SeparableConv2d(
               int(1024 * alpha),
               int(1024 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       self.fc = nn.Linear(int(1024 * alpha), class_num)
       self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def mobilenet(alpha=1, class_num=10):
    return MobileNet(alpha, class_num)

# check model summary
# "Model size" refers to the number of trainable parameters in the model?
# (3,32,32) -> grayscale (1,28,28)
net=mobilenet()
print(summary(net, (1,28,28)))