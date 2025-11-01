import torch
import torch.nn as nn
import torch.nn.functional as F


class block(nn.Module):
    def __init__(self, inc, outc):
        super(block, self).__init__()
        self.cv1 = nn.Conv2d(inc, outc, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(outc)
        self.relu1 = nn.ReLU(inplace = True)

        self.cv2 = nn.Conv2d(outc, outc, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.cv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.cv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class encoder_layer(nn.Module):
    def __init__(self, inc, outc):
        super(encoder_layer, self).__init__()
        self.b1 = block(inc, outc)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.b1.forward(x)
        skip = self.pool1(x)
        return skip, x

class decoder_layer(nn.Module):
    def __init__(self, inc, outc):
        super(decoder_layer, self).__init__()
        self.tp1 = nn.ConvTranspose2d(inc, inc//2, kernel_size = 2, stride = 2)
        self.b1 = block(inc, outc)

    def forward(self, skip, x):
        x = self.tp1(x)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, skip], dim = 1)
        x = self.b1.forward(x)
        return x

class ImprovedUNET(nn.Module):
    def __init__(self, classes = 8, channels = 1):
        super(ImprovedUNET, self).__init__()
        self.e1 = encoder_layer(1,64)
        self.e2 = encoder_layer(64,128)
        self.e3 = encoder_layer(128,256)
        self.e4 = encoder_layer(256,512)

        self.b1 = block(512,1024)

        self.d1 = decoder_layer(1024,512)
        self.d2 = decoder_layer(512,256)
        self.d3 = decoder_layer(256,128)
        self.d4 = decoder_layer(128,64)

        self.c1 = nn.Conv2d(64, classes, kernel_size = 1)

    def forward(self,x):
        skip1, x = self.e1(x)
        skip2, x = self.e2(x)
        skip3, x = self.e3(x)
        skip4, x = self.e4(x)

        x = self.b1(x)

        x = self.d1(skip4, x)
        x = self.d2(skip3, x)
        x = self.d3(skip2, x)
        x = self.d4(skip1, x)

        x = self.c1(x)

        return x