import os
import functools
from torchvision import transforms
from v2.util import conf


class ModelNet40Hyper:
    def __init__(self, data_root, v2_config):
        self.name = 'modelnet40'
        self.v2_config = v2_config
        self.c = functools.reduce(lambda a, b: int(a) * int(b), self.v2_config.split('_')[:4])

        self.data_root = data_root
        self.train_glob = os.path.join(data_root, '*/train/*/{}'.format(self.v2_config))
        self.test_glob = os.path.join(data_root, '*/test/*/{}'.format(self.v2_config))

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.all_cate = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf',
                         'bottle', 'bowl', 'car', 'chair', 'cone',
                         'cup', 'curtain', 'desk', 'door', 'dresser',
                         'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp',
                         'laptop', 'mantel', 'monitor', 'night_stand', 'person',
                         'piano', 'plant', 'radio', 'range_hood', 'sink',
                         'sofa', 'stairs', 'stool', 'table', 'tent',
                         'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

    def get_hyper_train(self):
        hyper_train_data = {
            'name': self.name,
            'data_dir': self.train_glob,
            'all_cate': self.all_cate,
            'transform': self.transform,
        }

        hyper_train_loader = {
            'batch_size': 128,
            'shuffle': True
        }
        return hyper_train_data, hyper_train_loader

    def get_hyper_test(self):
        hyper_test_data = {
            'name': self.name,
            'data_dir': self.test_glob,
            'all_cate': self.all_cate,
            'transform': self.transform,
        }

        hyper_test_loader = {
            'batch_size': 128,
            'shuffle': False
        }
        return hyper_test_data, hyper_test_loader

    def get_hyper_rst(self):
        hyper_rst = {
            'save': False,
            'rst_dir': './rst/cvpr/resnet_modelnet40_c%d_2.csv' % self.c
        }
        return hyper_rst
