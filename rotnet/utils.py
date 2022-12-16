from __future__ import print_function
import torch
import torch.utils.data as data

from torchvision import transforms
from torchvision.datasets import CIFAR10
import random
from torch.utils.data.dataloader import default_collate
from PIL import Image

import numpy as np



train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]
    )
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])



def rotate_img(img, rot):
    if rot == 0: # 0 degrees rotation
        return img
    elif rot ==1: # 90 degrees rotation
        return np.flipud(np.transpose(img, (1,0,2)))
    elif rot == 2: # 90 degrees rotation
        return np.fliplr(np.flipud(img))
    elif rot ==3: # 270 degrees rotation / or -90
        return np.transpose(np.flipud(img), (1,0,2))
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        index = random.randrange(0,4)
        img = rotate_img(img,  index)
        img = Image.fromarray(img)
        labels = torch.LongTensor([0, 1, 2, 3])
        if self.transform is not None:
            pos_1 = self.transform(img)
    
        target = labels[index]

        return pos_1, target