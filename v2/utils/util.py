import os
import glob
import math
from v2.utils import conf


def modelnet40_obj_unaligned():
    categories = sorted(os.listdir(conf.ModelNet40_OBJ))
    train_set = []
    test_set = []

    for ca in categories:
        train_objs = sorted(glob.glob(conf.ModelNet40_OBJ + '/{}/train/*.obj'.format(ca)))[:80]
        test_objs = sorted(glob.glob(conf.ModelNet40_OBJ + '/{}/test/*.obj'.format(ca)))[:20]
        train_set.extend(train_objs)
        test_set.extend(test_objs)

    return train_set, test_set


def modelnet40_aligned():
    categories = sorted(os.listdir(conf.ModelNet40_ALIGNED))
    train_set = []
    test_set = []

    for ca in categories:
        train_objs = sorted(glob.glob(conf.ModelNet40_ALIGNED + '/{}/train/*.off'.format(ca)))[:80]
        test_objs = sorted(glob.glob(conf.ModelNet40_ALIGNED + '/{}/test/*.off'.format(ca)))[:20]
        train_set.extend(train_objs)
        test_set.extend(test_objs)

    return train_set, test_set


def get_ca_set_id(path):
    """ A function to get:
        ca, category
        set, train or test
        id, object id

    Args:
        path: str
            The full path of the file

    Returns:

    """
    ca_ = path.split('/')[-3]
    set_ = path.split('/')[-2]
    id_ = os.path.splitext(os.path.basename(path))[0].split('_')[-1]

    return ca_, set_, id_


def get_factors(num):
    res = []
    for i in range(1, num // 2):
        if num % i == 0:
            res.append(i)
    res.append(num)
    return res


def get_perfect_factors(num):
    fac = get_factors(num)
    res = []
    for f in fac:
        if is_perfect_square(f) and is_perfect_square(num // f):
            res.append(f)
    return res


def is_perfect_square(num):
    return int(math.sqrt(num)) ** 2 == num
