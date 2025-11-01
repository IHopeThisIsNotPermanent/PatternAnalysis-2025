import torch
import torch.nn as nn


class block:
    def __init__(self, filters):
        self.filters = filters

        #TODO check what padding should be
        self.cv1 = nn.Conv2d(self.filters, 3, padding=2)
        #TODO check what feature num should be
        self.bn1 = nn.BatchNorm2d()
        self.relu1 = nn.ReLU()

        self.cv2 = nn.Conv2d(self.filters, 3, padding=2)
        self.bn2 = nn.BatchNorm2d()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.cv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.cv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x
    
class encoder_layer:
    def __init__(self, filters):
        self.b1 = block(self.filters)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.b1.forward(x)
        skip = self.pool1(x)
        return skip, x
    
class decoder_layer:
    def __init__(self, filters):
        self.tp1 = nn.ConvTranspose2d(filters, filters, (2,2))
        self.b1 = block(filters)
        
    def forward(self, skip, x):
        x = self.tp1(x)
        x = torch.cat([x, skip])
        x = self.b1.forward(x)
        return x

class ImprovedUNET(nn.Module):
    def __init__(self):
        super(ImprovedUNET, self).__init__()
        self.e1 = encoder_layer(64)
        self.e2 = encoder_layer(128)
        self.e3 = encoder_layer(256)
        self.e4 = encoder_layer(512)

        self.b1 = block(1024)

        self.d1 = decoder_layer(512)
        self.d2 = decoder_layer(256)
        self.d3 = decoder_layer(128)
        self.d4 = decoder_layer(64)

        self.c1 = nn.Conv2d(1,1,(2,2))

    def forward(self,x):
        skip1, x = self.e1.forward(x)
        skip2, x = self.e2.forward(x)
        skip3, x = self.e3.forward(x)
        skip4, x = self.e4.forward(x)

        x = self.b1(x)

        x = self.d1(x, skip1)
        x = self.d2(x, skip2)
        x = self.d3(x, skip3)
        x = self.d4(x, skip4)

        x = self.c1(x)

        return x