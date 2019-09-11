import numpy as np 
from easydict import EasyDict

__C = EasyDict()
cfg = __C

# config of point cloud pre-processing on Kitti
__C.KITTI = EasyDict()
__C.KITTI.PC_REDUCE_BY_RANGE = True
__C.KITTI.PC_AREA_SCOPE = np.array([[-40, 40],
                                    [-1,   3],
							        [0, 70.4]])  # x, y, z scope in rect camera coords

__C.KITTI.RANDOM_SELECT = True #random select num_points from raw point cloud
__C.KITTI.NUM_POINTS = 16384
__C.KITTI.CLASSES = 'Car, Pedestrian, Cyclist'

__C.KITTI.USE_INTENSITY = True

# config of augmentation
__C.KITTI.AUG_DATA = True
__C.KITTI.AUG_METHOD_LIST = ['rotation', 'scaling', 'flip']
__C.KITTI.AUG_METHOD_PROB = [0.5, 0.5, 0.5]
__C.KITTI.MUST_AUG = ['rotation', 'scaling']
__C.KITTI.AUG_ROT_RANGE = 18


# config of point cloud pre-processing on SUNRGBD
__C.SUNRGBD = EasyDict()
__C.SUNRGBD.MODES = ['train', 'valid', 'test', 'trainvalid']
__C.SUNRGBD.CLASSES =  ['bed','table','sofa','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub']
# __C.SUNRGBD.CLASS2INDEX = {'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9}
__C.SUNRGBD.CLASS2INDEX = {'__background__':0, 'bed':1, 'table':2, 'sofa':3, 'chair':4, 'toilet':5, 'desk':6, 'dresser':7, 'night_stand':8, 'bookshelf':9, 'bathtub':10}
__C.SUNRGBD.INDEX2CLASS = {str(__C.SUNRGBD.CLASS2INDEX[k]): k for k in __C.SUNRGBD.CLASS2INDEX}
__C.SUNRGBD.MIN_NUM_FOV_POINTS = 512


# config on ScanNet
__C.SCANNET = EasyDict()
__C.SCANNET.CLASSES = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter',
                       'desk', 'curtain', 'refridgerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
__C.SCANNET.CLASS2INDEX = {cls:idx for idx, cls in enumerate(__C.SCANNET.CLASSES)}
__C.SCANNET.IMAGE_WIDTH = 1296
__C.SCANNET.IMAGE_HEIGHT = 968
__C.SCANNET.DEPTH_WIDTH = 640
__C.SCANNET.DEPTH_HEIGHT = 480

