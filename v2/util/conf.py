import os

HOME_DIR = os.getenv('HOME')
MEDIA_DATA_DIR = '/media/tengyu/DataU'
DATA_DIR = os.path.join(MEDIA_DATA_DIR, 'Data')
PJ_ROOT = os.path.dirname(os.path.abspath(__file__))

CKPT_DIR = os.path.join(PJ_ROOT, 'exp', 'ckpt')
RST_DIR = os.path.join(PJ_ROOT, 'exp', 'rst')

ModelNet_DIR = os.path.join(DATA_DIR, 'ModelNet')
ModelNet10_DIR = os.path.join(ModelNet_DIR, 'ModelNet10')
ModelNet40_DIR = os.path.join(ModelNet_DIR, 'ModelNet40')
ModelNet10FacesNP_DIR = os.path.join(ModelNet_DIR, 'ModelNet10FacesNP')
ModelNet40FacesNP_DIR = os.path.join(ModelNet_DIR, 'ModelNet40FacesNP')
ModelNet10_c10000_DIR = os.path.join(ModelNet_DIR, 'ModelNet10_c10000')
ModelNet40_c10000_DIR = os.path.join(ModelNet_DIR, 'ModelNet40_c10000')
ModelNet40_c22500_DIR = os.path.join(ModelNet_DIR, 'ModelNet40_c22500')
ModelNet40_c40000_DIR = os.path.join(ModelNet_DIR, 'ModelNet40_c40000')
