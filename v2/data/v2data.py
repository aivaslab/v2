import os
import glob
import pickle
import torch
import torchvision
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from v2.data.v2gen import V2Generator


class V2Off(Dataset):
    def __init__(self, name, files, all_cate, transform, v2_config):
        """ Offline V2 representation dataset

        Args:
            name:
            files:
            all_cate:
            transform:
            v2_config:
        """
        self.name = name
        self.all_cate = all_cate
        self.v2_config = v2_config

        self.files = files
        self.labels = list(map(self._get_label, self.files))
        self.transform = transform

    def _get_label(self, dir_):
        return self.all_cate.index(dir_.split('/')[-4])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_dir = self.files[index]
        with open(img_dir, 'rb') as f:
            x = pickle.load(f)

        # x = np.dstack([x, x, x])
        # x = np.expand_dims(x, axis=-1)
        if self.transform:
            x = self.transform(x)
            x = x.to(torch.float)

        x[0, :, :] = x[0, :, :] - 1
        x[1, :, :] = x[1, :, :] * 2 - 1
        x[2, :, :] = x[2, :, :] * 2 - 1
        x[3, :, :] = x[3, :, :] - 1
        x[4, :, :] = x[4, :, :] * 2 - 1
        x[5, :, :] = x[5, :, :] * 2 - 1

        y = self.labels[index]

        # import matplotlib.pyplot as plt
        # a = x[0, :, :]
        # plt.imshow(a, cmap='gray')
        # plt.show()
        return x, y


class V2On(Dataset):
    def __init__(self, name, files, all_cate, transform, v2_config):
        """ Offline V2 representation dataset

        Args:
            name:
            files,
            all_cate:
            transform:
            v2_config:
        """
        self.name = name
        self.all_cate = all_cate
        self.v2_config = v2_config

        self.files = files
        self.labels = list(map(self._get_label, self.files))
        self.transform = transform

        m = 128
        n = 128
        h = 1
        w = 1
        d = 1 / 128
        r = 1
        polar = False
        ssm = 's2cnn'  # sphere sampling method
        v2m = 'mt'
        rot = True
        rtr = 0.1

        self.v2generator = V2Generator(m, n, w, h, d, r, polar, ssm, v2m, rot, rtr)

    def _get_label(self, dir_):
        return self.all_cate.index(dir_.split('/')[-3])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        obj = self.files[index]
        self.v2generator(obj)
        x = self.v2generator.v2

        if self.transform:
            x = self.transform(x)
            x = x.to(torch.float)

        y = self.labels[index]
        return x, y

