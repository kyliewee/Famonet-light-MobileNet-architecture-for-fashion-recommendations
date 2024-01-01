#import torch
import torch.nn as nn
from torchsummary import summary

class DepthwiseSeparableConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):

        super().__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(
            input_channels,
            input_channels,
            kernel_size,
            groups = input_channels,
            **kwargs
            ),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace = True)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(
            input_channels, 
            output_channels, 
            kernel_size = 1
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):

        x = self.depthwise(x)
        x = self.pointwise(x)

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

class ModelV1(nn.Module):

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
           DepthwiseSeparableConv2d(
           int(32 * alpha),
           int(64 * alpha),
           kernel_size = 3,
           padding = 1,
           bias = False
           )
       )

       self.conv1 = nn.Sequential(
           DepthwiseSeparableConv2d(
           int(64 * alpha),
           int(128 * alpha),
           kernel_size = 3,
           stride = 2,
           padding = 1,
           bias = False
           ),
           DepthwiseSeparableConv2d(
           int(128 * alpha),
           int(128 * alpha),
           kernel_size = 3,
           padding = 1,
           bias = False
           )
       )

       self.conv2 = nn.Sequential(
           DepthwiseSeparableConv2d(
           int(128 * alpha),
           int(256 * alpha),
           kernel_size = 3,
           stride = 2,
           padding = 1,
           bias = False
           ),
           DepthwiseSeparableConv2d(
           int(256 * alpha),
           int(256 * alpha),
           kernel_size = 3,
           padding = 1,
           bias = False
           )
       )

       self.conv3 = nn.Sequential(
           DepthwiseSeparableConv2d(
           int(256 * alpha),
           int(512 * alpha),
           kernel_size = 3,
           stride = 2,
           padding = 1,
           bias = False
           ),
           DepthwiseSeparableConv2d(
               int(512 * alpha),
               int(512 * alpha),
               kernel_size = 3,
               padding = 1,
               bias = False
           ),
           DepthwiseSeparableConv2d(
               int(512 * alpha),
               int(512 * alpha),
               kernel_size = 3,
               padding = 1,
               bias = False
           ),
           DepthwiseSeparableConv2d(
               int(512 * alpha),
               int(512 * alpha),
               kernel_size = 3,
               padding = 1,
               bias = False
           ),
           DepthwiseSeparableConv2d(
               int(512 * alpha),
               int(512 * alpha),
               kernel_size = 3,
               padding = 1,
               bias=False
           ),
           DepthwiseSeparableConv2d(
               int(512 * alpha),
               int(512 * alpha),
               kernel_size = 3,
               padding = 1,
               bias = False
           )
       )

       self.conv4 = nn.Sequential(
           DepthwiseSeparableConv2d(
               int(512 * alpha),
               int(1024 * alpha),
               kernel_size = 3,
               stride = 2,
               padding = 1,
               bias = False
           ),
           DepthwiseSeparableConv2d(
               int(1024 * alpha),
               int(1024 * alpha),
               kernel_size = 3,
               padding = 1,
               bias = False
           )
       )

       self.fc = nn.Linear(int(1024 * alpha), class_num)
       self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):

        x = self.stem(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


# check model summary
def Model(class_num = 10, alpha = 1):
    return ModelV1(class_num, alpha)

model = Model()
print(summary(model, (1,28,28)))
print(model)
