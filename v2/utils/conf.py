import os

HOME_DIR = os.getenv('HOME')
MEDIA_DATA_DIR = '/media/tengyu/DataU'
DATA_DIR = os.path.join(MEDIA_DATA_DIR, 'Data')
PJ_ROOT = '/home/tengyu/Documents/research/v2/v2'

EXP_DIR = os.path.join(PJ_ROOT, 'exp')
PROJ_DATA_DIR = os.path.join(PJ_ROOT, 'data')
CKPT_DIR = os.path.join(EXP_DIR, 'ckpt')
RST_DIR = os.path.join(EXP_DIR, 'res')

ModelNet = os.path.join(DATA_DIR, 'ModelNet')

ModelNet40 = os.path.join(ModelNet, 'ModelNet40')

ModelNet40_ALIGNED = os.path.join(ModelNet40, 'ModelNet40_ALIGNED')
ModelNet40_NP = os.path.join(ModelNet40, 'ModelNet40_NP')
ModelNet40_OBJ = os.path.join(ModelNet40, 'ModelNet40_OBJ')

ModelNet40_D = os.path.join(ModelNet40, 'D')
ModelNet40_DSC = os.path.join(ModelNet40, 'DSC')
ModelNet40_DSCDSC = os.path.join(ModelNet40, 'DSCDSC')
