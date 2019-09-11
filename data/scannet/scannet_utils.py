# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Ref: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts
    Modified by Zhao Na, Sep 2019
'''

import os
import sys
import json
import pickle
import csv

try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)

try:
    from plyfile import PlyData, PlyElement
except:
    print("Please install the module 'plyfile' for PLY i/o, e.g.")
    print("pip install plyfile")
    sys.exit(-1)

def represents_int(s):
    ''' if string s represents an int. '''
    try: 
        int(s)
        return True
    except ValueError:
        return False


def read_label_mapping(filename, label_from='raw_category', label_to='nyu40class'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = str(row[label_to])
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k):v for k,v in mapping.items()}
    return mapping

def read_mesh_vertices(filename):
    """ read XYZ for each vertex.
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
    return vertices

def read_mesh_vertices_rgb(filename):
    """ read XYZ RGB for each vertex.
    Note: RGB values are in 0-255
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']
    return vertices


def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_label = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId']+1 # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            object_id_to_label[object_id] = label
    return object_id_to_label


def read_camera_intrinsic(file_path):
    '''Read camera intrinsic parameters from meta file (<sceneid>.txt)
        [[fx, 0,  mx, 0]
         [0,  fy, my, 0]
         [0,  0,  1,  0]]
    '''
    lines = open(file_path).readlines()
    for line in lines:
        if 'fx_color' in line:
            fx = float(line.rstrip().strip('fx_color = '))
        elif 'fy_color' in line:
            fy = float(line.rstrip().strip('fy_color = '))
        elif 'mx_color' in line:
            mx = float(line.rstrip().strip('mx_color = '))
        elif 'my_color' in line:
            my = float(line.rstrip().strip('my_color = '))
    camera_intrinsic = np.zeros((3,4), dtype=np.float32)
    try:
        camera_intrinsic[0,0] = fx
        camera_intrinsic[0,2] = mx
        camera_intrinsic[1,1] = fy
        camera_intrinsic[1,2] = my
    except:
        raise('Error! Camera intrinsic parameters are not completely extracted!\n '
                                    '[fx:{0},fy:{1},mx:{2},my:{3}]'.format(fx, fy, mx, my))
    camera_intrinsic[2,2] = 1.
    return camera_intrinsic


def read_depth_intrinsic(file_path):
    '''Read depth intrinsic parameters from meta file (<sceneid>.txt)
        [[fx, 0,  mx, 0]
         [0,  fy, my, 0]
         [0,  0,  1,  0]]
    '''
    lines = open(file_path).readlines()
    for line in lines:
        if 'fx_depth' in line:
            fx = float(line.rstrip().strip('fx_depth = '))
        elif 'fy_depth' in line:
            fy = float(line.rstrip().strip('fy_depth = '))
        elif 'mx_depth' in line:
            mx = float(line.rstrip().strip('mx_depth = '))
        elif 'my_depth' in line:
            my = float(line.rstrip().strip('my_depth = '))
    depth_intrinsic = np.zeros((3,4), dtype=np.float32)
    try:
        depth_intrinsic[0,0] = fx
        depth_intrinsic[0,2] = mx
        depth_intrinsic[1,1] = fy
        depth_intrinsic[1,2] = my
    except:
        raise('Error! Depth intrinsic parameters are not completely extracted!\n '
                                    '[fx:{0},fy:{1},mx:{2},my:{3}]'.format(fx, fy, mx, my))
    depth_intrinsic[2,2] = 1.
    return depth_intrinsic



def write_2Dbbox(save_dir, frame_idx, bbox2d):
    save_path = os.path.join(save_dir, '{0}_bbox.pkl'.format(frame_idx))
    with open(save_path, 'wb') as f:
        pickle.dump(bbox2d, f)

def read_2Dbbox(file_path):
    with open(file_path, 'rb') as f:
        bbox2d = pickle.load(f)
    return bbox2d