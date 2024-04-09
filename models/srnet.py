import torch
import torch.nn as nn
from .SRNet.model import Srnet

class SRNet(nn.Module):
    def __init__(
            self,
            patch_size=256,
            num_classes=0,
            **args,
        ):
        super().__init__()
        self.srnet = Srnet()
        self.num_classes = num_classes
        if self.num_classes > 0:
            self.output = torch.nn.Linear(512, self.num_classes)
            
    def forward(self, x):
        x = self.srnet(x)
        if self.num_classes > 0:
            x = self.output(x)
        return x
