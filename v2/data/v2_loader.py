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
    # c10000
    cam_settings1 = ['4_1_50_50_0.02', '8_2_25_25_0.02', '20_5_10_10_0.02', '40_10_5_5_0.02', '100_25_2_2_0.02', '200_50_1_1_0.02']

    # c22500
    cam_settings2 = ['4_1_75_75_0.01', '12_3_25_25_0.01', '20_5_15_15_0.01', '60_15_5_5_0.01', '100_25_3_3_0.01', '300_75_1_1_0.01']

    # c40000
    cam_settings3 = ['4_1_100_100_0.01', '8_2_50_50_0.01', '16_4_25_25_0.01', '20_5_20_20_0.01',
                     '40_10_10_10_0.01', '80_20_5_5_0.01', '100_25_4_4_0.01', '200_50_2_2_0.01', '400_100_1_1_0.01']

    cams = cam_settings1 + cam_settings2 + cam_settings3

    for cam in cams:
        v2loader = V2Loader(ModelNet40Hyper(cam))



if __name__ == '__main__':
    main()
