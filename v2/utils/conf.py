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

ModelNet40_ALLCATE = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf',
                      'bottle', 'bowl', 'car', 'chair', 'cone',
                      'cup', 'curtain', 'desk', 'door', 'dresser',
                      'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp',
                      'laptop', 'mantel', 'monitor', 'night_stand', 'person',
                      'piano', 'plant', 'radio', 'range_hood', 'sink',
                      'sofa', 'stairs', 'stool', 'table', 'tent',
                      'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
