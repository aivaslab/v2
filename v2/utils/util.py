import os
import glob
from v2.utils import conf


def modelnet40_objs():
    categories = sorted(os.listdir(conf.ModelNet40OBJ_DIR))
    train_set = []
    test_set = []

    for ca in categories:
        train_objs = sorted(glob.glob(conf.ModelNet40OBJ_DIR + '/{}/train/*.obj'.format(ca)))[:80]
        test_objs = sorted(glob.glob(conf.ModelNet40OBJ_DIR + '/{}/test/*.obj'.format(ca)))[:20]
        train_set.extend(train_objs)
        test_set.extend(test_objs)

    return train_set, test_set


def get_dst(obj, old, new, v2_config):
    dst_ = obj.replace(old, new)
    dir_ = os.path.dirname(dst_)
    fname = os.path.splitext(os.path.basename(dst_))[0].split('_')
    # In some cases the category name contains '_' such as flower_pot_0001.obj
    ca_, id_ = '_'.join(fname[:-1]), fname[-1]

    dst = os.path.join(dir_, id_, v2_config)
    dir_ = os.path.dirname(dst)
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    return dst
