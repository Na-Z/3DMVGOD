# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Ref: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts
    Modified by Zhao Na, Sep 2019
'''

import os
import sys
import math
import json
import pickle
import cv2
import csv

from config import cfg

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


def read_color2depth_extrinsic(file_path):
    '''Read colorToDepthExtrinsics matrix'''
    lines = open(file_path).readlines()
    for line in lines:
        if 'colorToDepthExtrinsics' in line:
            color2depth_extrinsic = [float(x) \
                                     for x in line.rstrip().strip('colorToDepthExtrinsics = ').split(' ')]
            break
    else:
        return None
    color2depth_extrinsic = np.array(color2depth_extrinsic).reshape((4, 4))
    return color2depth_extrinsic


def read_2Dbbox(file_path):
    with open(file_path, 'rb') as f:
        bbox2d = pickle.load(f)
    return bbox2d


def write_2Dbbox(save_dir, frame_idx, bbox2d):
    save_path = os.path.join(save_dir, '{0}_bbox.pkl'.format(frame_idx))
    with open(save_path, 'wb') as f:
        pickle.dump(bbox2d, f)


###################################### DEPTH MAPS TO POINT CLOUDS #################################

def _parse_intrinsic_matrix(intrinsic):
    '''
    :param intrinsic: np.ndarray, shape (3,4)
            [[fx, 0,  mx, 0]
             [0,  fy, my, 0]
             [0,  0,  1,  0]]
    :return: fx, fy: focal length along x and y axis, respectively
             mx, my: distance to the principle point in terms of x and y axis, respectively
    '''
    fx = intrinsic[0,0]
    fy = intrinsic[1,1]
    mx = intrinsic[0,2]
    my = intrinsic[1,2]

    return fx, fy, mx, my


def _rescale_bbox_dim(bbox2d_dimension):
    '''
    recale the bbox dimension in the color image to the depth image
    :param bbox2d_dimension: [x_min, y_min, x_max, y_max] in the color image
    :return: corresponding dimension in the depth image
    '''
    depth_x_min = math.floor(bbox2d_dimension[0] * cfg.SCANNET.DEPTH_WIDTH / cfg.SCANNET.IMAGE_WIDTH)
    depth_y_min = math.floor(bbox2d_dimension[1] * cfg.SCANNET.DEPTH_HEIGHT / cfg.SCANNET.IMAGE_HEIGHT)
    depth_x_max = min(math.ceil(bbox2d_dimension[2] * cfg.SCANNET.DEPTH_WIDTH / cfg.SCANNET.IMAGE_WIDTH),
                      cfg.SCANNET.DEPTH_WIDTH-1)
    depth_y_max = min(math.ceil(bbox2d_dimension[3] * cfg.SCANNET.DEPTH_HEIGHT / cfg.SCANNET.IMAGE_HEIGHT),
                      cfg.SCANNET.DEPTH_HEIGHT-1)

    return depth_x_min, depth_y_min, depth_x_max, depth_y_max


def _depth_map_to_points(sel_depth_map, depth_intrinsic, cols, rows):
    '''
    Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    :param cols: numpy array, contains the index of each selected column
    :param rows: numpy array, contains the index of each selected row
    '''
    fx, fy, mx, my = _parse_intrinsic_matrix(depth_intrinsic)

    valid = (sel_depth_map > 0)
    z = np.where(valid, sel_depth_map / cfg.SCANNET.DEPTH_SCALING_FACTOR, np.nan)
    x = np.where(valid, z * (cols - mx) / fx, 0)
    y = np.where(valid, z * (rows - my) / fy, 0)

    return x, y, z, valid


def depth_to_point_cloud(depth_img, depth_intrinsic):
    '''
    depth_image is a 2-D ndarray with shape (rows, cols) containing
    depths from 1 to 254 inclusive. The result is a 3-D array with
    shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.

    :param depth_img: np.ndarray, size (480*640)
           depth_intrinsic: np.ndarray, shape (3,4)
    '''
    rows, cols = depth_img.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    x, y, z, _ = _depth_map_to_points(depth_img, depth_intrinsic, c, r)

    pts = np.dstack((x, y, z)).reshape(-1, 3)
    ptcloud = pts[~np.isnan(pts[:,2]), :]

    return ptcloud


def cropped_depth_to_point_cloud(depth_img, depth_intrinsic, bbox2d_dimension):
    '''
   The depth_image is cropped using the bbox annotation in color image.
    The result is a 3-D array with shape (bbox_rows, bbox_cols, 3).
    Pixels with invalid depth in the input have NaN for the z-coordinate
    in the result.

    :param depth_img: np.ndarray, size (480*640)
           depth_intrinsic: np.ndarray, shape (3,4)
           bbox2d_dimension: list, [x_min, y_min, x_max, y_max]

    '''
    xmin, ymin, xmax, ymax = _rescale_bbox_dim(bbox2d_dimension)

    depth_img_crop = depth_img[ymin:ymax, xmin:xmax]

    c, r = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax), sparse=True)
    x, y, z, _ = _depth_map_to_points(depth_img_crop, depth_intrinsic, c, r)

    pts = np.dstack((x, y, z)).reshape(-1, 3)
    ptcloud = pts[~np.isnan(pts[:, 2]), :]

    return ptcloud


def cropped_depth_to_point_cloud_with_cam(depth_img, depth_intrinsic, bbox2d_dimension, CAM):
    '''
    The depth_image is cropped using the bbox annotation in color image.
    Also, the cam is also cropped accordingly and concatenate with the
    coordinates restored from cropped depth image.
    The result is a 3-D array with shape (bbox_rows, bbox_cols, 4).
    Pixels with invalid depth in the input have NaN for the z-coordinate in the result.
    The forth dimension indicates the activation score with respect to the bbox class.

    :param depth_img: np.ndarray, size (480*640)
           depth_intrinsic: np.ndarray, shape (3,4)
           bbox2d_dimension: list, [x_min, y_min, x_max, y_max]
           CAM: np.ndarray. shape (7, 10)
    '''
    xmin, ymin, xmax, ymax = _rescale_bbox_dim(bbox2d_dimension)
    #TODO: decide which interpolation type to be used in upsampling CAM
    cam = cv2.resize(CAM, (cfg.SCANNET.DEPTH_WIDTH, cfg.SCANNET.DEPTH_HEIGHT)) #(480*640)

    depth_img_crop = depth_img[ymin:ymax, xmin:xmax]
    cam_crop = cam[ymin:ymax, xmin:xmax]

    c, r = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax), sparse=True)
    x, y, z, valid = _depth_map_to_points(depth_img_crop, depth_intrinsic, c, r)

    s = np.where(valid, cam_crop, np.nan) #activation_score

    pts_cam = np.dstack((x, y, z, s)).reshape(-1, 4)
    ptcloud_cam = pts_cam[~np.isnan(pts_cam[:, 2]), :]

    return ptcloud_cam

#####################################################################################################


################################# Coordinate Frames Transformation  #################################

def project_image_to_camera(pts, K):
    '''
    Project the color/depth image frame to its corresponding camera frame
    :param pts: numpy array (n,3)
    :param K: depth/color camera intrinsic parameters
    :return: pts_camera: numpy array (n,3), points in depth/color camera coordinate
    '''
    fx, fy, mx, my = _parse_intrinsic_matrix(K)

    z = pts[:, 2] / cfg.SCANNET.DEPTH_SCALING_FACTOR
    x = z * (pts[:, 0] - mx) / fx
    y = z * (pts[:, 1] - my) / fy

    n = pts.shape[0]
    pts_camera = np.zeros((n, 3))
    pts_camera[:, 0] = x
    pts_camera[:, 1] = y
    pts_camera[:, 2] = z

    return pts_camera


def calibrate_camera_depth_to_color(pts, K):
    '''
    Project the depth camera frame to the color camera frame
    :param pts: numpy array (n,3), points in depth camera coordinate
    :param K: color2depth camera intrinsic parameters
    :return: pts_color: numpy array (n,3), points in color camera coordinate
    '''
    pts_color = np.ones((pts.shape[0], 4))
    pts_color[:, 0:3] = pts[:, 0:3]

    if K is not None:
        pts_color = np.dot(pts_color, K.transpose())

    return pts_color

#####################################################################################################


