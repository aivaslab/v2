import os
import itertools
import numpy as np
import matplotlib.pylab as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, inception_v3, vgg11

from v2.data import v2_loader
from v2.exp.hyper_p import ModelNet40Hyper
from v2.exp.run_exp import Learner, save_confusion_matrix, save_results
from v2.util import conf


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    cudnn.benchmark = True


def main():
    data_root = conf.ModelNet40_MDSC_CDSC_C16384
    m = 128
    n = 128
    h = 1
    w = 1
    d = 1 / 128
    v2_config = '{}_{}_{}_{}_{:.4f}.npy'.format(m, n, h, w, d)

    # Data
    v2loader = v2_loader.V2Loader(ModelNet40Hyper(data_root, v2_config))

    # Model
    net = resnet18()
    net.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.fc = nn.Linear(512, 40)
    net = net.to(DEVICE)

    learner = Learner(net, v2loader.train_loader, v2loader.test_loader)
    # for epoch in range(1000):
    #     loss_tr, acc_tr = learner.train(epoch)
    #     if epoch % 100 == 0:
    #         loss_te, acc_te = learner.test(epoch)

    # Final Test
    learner.final_test()


if __name__ == '__main__':
    main()
