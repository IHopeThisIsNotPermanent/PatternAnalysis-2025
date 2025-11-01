import torch
import torch.nn as nn




class Encoder(nn.Module):
    def __init__(self):
        pass
    def forward(self,x):
        return x



class Decoder(nn.Module):
    def __init__(self):
        pass
    def forward(self,x):
        return x

class ImprovedUNET(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(ImprovedUNET, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
    def forward(self,x):
        encoder_outs = self.Encoder(x)
        x = self.Decoder(encoder_outs)
        return x