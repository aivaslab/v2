import os
import glob
import pickle
import torch
import torchvision
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from v2.data.v2gen import V2Generator


class V2(Dataset):
    def __init__(self, files, all_cate, transform):
        """ Offline V2 representation dataset

        Args:
            files:
            all_cate:
            transform:
        """
        self.all_cate = all_cate

        self.files = files
        self.labels = list(map(self._get_label, self.files))
        self.transform = transform

    def _get_label(self, dir_):
        return self.all_cate.index(dir_.split('/')[-5])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_dir = self.files[index]
        with open(img_dir, 'rb') as f:
            x = pickle.load(f)

        if self.transform:
            x = x[:, :, 0]
            x = self.transform(x)
            x = x.to(torch.float)

        y = self.labels[index]
        return x, y
