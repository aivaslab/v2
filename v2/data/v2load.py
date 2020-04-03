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
from v2.utils import conf
from v2.exp.hyper_p import ModelNet40Hyper
from v2.data.v2data import V2Off, V2On


class V2Loader:
    def __init__(self, hyper):
        if hyper.online:
            self.V2 = V2On
        else:
            self.V2 = V2Off

        self.hyper = hyper
        self.train_hyper = hyper.get_hyper_train()
        self.test_hyper = hyper.get_hyper_test()

        self.train_set, self.train_loader = self._loader(*self.train_hyper)
        self.test_set, self.test_loader = self._loader(*self.test_hyper)

    def _loader(self, hyper_p_data, hyper_p_loader):
        data_set = self.V2(**hyper_p_data)
        data_loader = DataLoader(data_set, **hyper_p_loader)
        return data_set, data_loader


def main():
    data_root = conf.ModelNet40_MDSC_CDSC_C16384
    m = 128
    n = 128
    h = 1
    w = 1
    d = 1 / 128
    v2_config = '{}_{}_{}_{}_{:.4f}.npy'.format(m, n, h, w, d)

    v2loader = V2Loader(ModelNet40Hyper(data_root, v2_config, online=True))


if __name__ == '__main__':
    main()
