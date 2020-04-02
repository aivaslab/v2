import os

HOME_DIR = os.getenv('HOME')
MEDIA_DATA_DIR = '/media/tengyu/DataU'
DATA_DIR = os.path.join(MEDIA_DATA_DIR, 'Data')
PJ_ROOT = '/home/tengyu/Documents/research/v2/v2'

CKPT_DIR = os.path.join(PJ_ROOT, 'exp', 'ckpt')
RST_DIR = os.path.join(PJ_ROOT, 'exp', 'rst')

ModelNet_DIR = os.path.join(DATA_DIR, 'ModelNet')
ModelNet40_DIR = os.path.join(ModelNet_DIR, 'ModelNet40')
ModelNet40OBJ_DIR = os.path.join(ModelNet_DIR, 'ModelNet40_OBJ')

ModelNet40_c10000_DIR = os.path.join(ModelNet_DIR, 'ModelNet40_c10000')
ModelNet40_c22500_DIR = os.path.join(ModelNet_DIR, 'ModelNet40_c22500')
ModelNet40_c40000_DIR = os.path.join(ModelNet_DIR, 'ModelNet40_c40000')

ModelNet40_MDSC_CDSC = os.path.join(ModelNet_DIR, 'mdsc_cdsc')  # mesh dsc channels and convex hull dsc channels
ModelNet40_MDSC_CDSC_C16384 = os.path.join(ModelNet40_MDSC_CDSC, 'C16384')
