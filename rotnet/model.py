import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head

        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return  F.normalize(out, dim=-1)


# class BasicBlock(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1):
#         super(BasicBlock, self).__init__()
#         padding = (kernel_size-1)//2
#         self.layers = nn.Sequential()
#         self.layers.add_module('Conv', nn.Conv2d(in_planes, out_planes, \
#             kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
#         self.layers.add_module('BatchNorm', nn.BatchNorm2d(out_planes))
#         self.layers.add_module('ReLU',      nn.ReLU(inplace=True))

#     def forward(self, x):
#         return self.layers(x)
# class GlobalAvgPool(nn.Module):
#     def __init__(self):
#         super(GlobalAvgPool, self).__init__()

#     def forward(self, feat):
#         assert(feat.size(2) == feat.size(3))
#         feat_avg = F.avg_pool2d(feat, feat.size(2)).view(-1, feat.size(1))
#         return feat_avg
# class Model(nn.Module):
#     def __init__(self, feature_dim=128):
#         super(Model, self).__init__()

#         self.f = []
#         for name, module in resnet50().named_children():
#             if name == 'conv1':
#                 module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#             if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
#                 self.f.append(module)
#         # encoder
#         self.f = nn.Sequential(*self.f)
#         # projection head
#         nchannels = 10
      
#         self.g = nn.Sequential()
#         self.g.add_module('Block3_ConvB1',  BasicBlock(nchannels, 192, 3))
#         self.g.add_module('Block3_ConvB2',  BasicBlock(192, 192, 1))
#         self.g.add_module('Block3_ConvB3',  BasicBlock(192, 192, 1))
#         self.g.add_module('GlobalAvgPool',  GlobalAvgPool())
#         self.g.add_module('Liniear_F',      nn.Linear(192, 4))
#     def forward(self, x):
#         x = self.f(x)
#         feature = x
#         print(x.shape)
#         out = self.g(feature)
#         return  F.normalize(out, dim=-1)