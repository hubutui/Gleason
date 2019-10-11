#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset
import os.path as osp
import os
from PIL import Image


class Gleason(Dataset):
    def __init__(self, imgdir, maskdir=None, train=True, val=False,
                 test=False, transforms=None, transform=None, target_transform=None):
        super(Gleason, self).__init__()
        self.imgdir = imgdir
        self.maskdir = maskdir
        self.imglist = sorted(os.listdir(imgdir))
        if not test:
            self.masklist = [item.replace('.jpg', '_classimg_nonconvex.png') for item in self.imglist]
        else:
            self.masklist = []
        self.train = train
        self.val = val
        self.test = test
        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        image = Image.open(osp.join(self.imgdir, self.imglist[idx]))
        if not self.test:
            mask = Image.open(osp.join(self.maskdir, self.masklist[idx]))
        if self.transforms and not self.test:
            image, mask = self.transforms(image, mask)
        if self.transform:
            image = self.transform(image, mask)
        if self.target_transform and not self.test:
            mask = self.target_transform(mask)

        if self.test:
            return image
        else:
            return image, mask
