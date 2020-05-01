import os
import glob
import math
import imageio
import numpy as np

from v2.utils import conf
from PIL import Image


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


def flatten(l):
    return [item for sublist in l for item in sublist]


def trim_boundary(array, trim_n, m, n):
    array = np.split(array.reshape(int(np.sqrt(array.size)), int(np.sqrt(array.size))), m, axis=0)
    array = list(map(lambda x_: np.split(x_, n, axis=1), array))
    array = flatten(array)

    if type(trim_n) == int:
        array = list(map(lambda x_: x_[trim_n: -trim_n, trim_n: -trim_n], array))
    elif type(trim_n) == list:
        assert len(trim_n) == 4, 'Invalid trimming number {}.'.format(len(trim_n))
        array = list(map(lambda x_: x_[trim_n[0]: -trim_n[1], trim_n[2]: -trim_n[3]], array))

    return array


def make_gif(dir_, to_dir_):
    files = sorted(glob.glob(dir_), key=lambda x: int(os.path.basename(x).split('.')[0]))

    img, *imgs = [Image.open(f).convert('RGB') for f in files][::-1]

    img.save(fp=to_dir_, append_images=imgs,
             save_all=True, duration=200, loop=0)


def update_resolution(dir_, to_dir_):
    files = glob.glob(dir_)
    imgs = [Image.open(f).resize((1024, 1024), Image.ANTIALIAS) for f in files]
    for i, img in enumerate(imgs):
        img.save(fp=os.path.join(to_dir_, os.path.basename(files[i])))


def img_crop(dir_):
    files = sorted(glob.glob(dir_))

    for file in files:
        img = imageio.imread(file)
        cropping = 1300
        offset = 150
        img = img[cropping:-cropping, cropping+offset:-cropping+offset]
        imageio.imsave(file.replace('org', 'cropped'), img)
