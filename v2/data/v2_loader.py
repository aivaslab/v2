import os
import glob
import pickle
import torch
import torchvision
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import matplotlib.pyplot as plt
from v2.util import conf
from v2.exp.hyper_p import ModelNet40Hyper


class V2(Dataset):
    def __init__(self, name, data_dir, all_cate, transform):
        self.name = name
        self.data_dir = data_dir
        self.all_cate = all_cate

        self.files = sorted(glob.glob(data_dir))
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

        y = self.labels[index]
        return x, y


class V2Loader:
    def __init__(self, hyper):
        self.hyper = hyper
        self.train_hyper = hyper.get_hyper_train()
        self.test_hyper = hyper.get_hyper_test()

        self.train_loader = self._loader(*self.train_hyper)
        self.test_loader = self._loader(*self.test_hyper)

    def _loader(self, hyper_p_data, hyper_p_loader):
        data_loader = DataLoader(V2(**hyper_p_data), **hyper_p_loader)
        return data_loader


def main():
    data_root = conf.ModelNet40_MDSC_CDSC_C16384
    m = 128
    n = 128
    h = 1
    w = 1
    d = 1 / 128
    v2_config = '{}_{}_{}_{}_{:.4f}.npy'.format(m, n, h, w, d)

    v2loader = V2Loader(ModelNet40Hyper(data_root, v2_config))


if __name__ == '__main__':
    main()
