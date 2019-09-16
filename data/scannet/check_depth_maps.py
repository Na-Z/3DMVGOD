import os
import random
import numpy as np
import cv2
from PIL import Image
import json
from zipfile import ZipFile
import matplotlib.pyplot as plt
from mayavi import mlab
from vtk_visualizer.plot3d import *
from vtk_visualizer import get_vtk_control

import scannet_utils
from config import cfg


def depth_to_point_cloud(depth_img, depth_intrinsic):
    '''
    Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    depth_image is a 2-D ndarray with shape (rows, cols) containing
    depths from 1 to 254 inclusive. The result is a 3-D array with
    shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.

    :param depth_img: np.ndarray, size (480*640)
           depth_intrinsic: np.ndarray, shape (4*3)
            [[fx, 0,  mx, 0]
             [0,  fy, my, 0]
             [0,  0,  1,  0]]

    '''
    rows, cols = depth_img.shape
    fx = depth_intrinsic[0,0]
    fy = depth_intrinsic[1,1]
    mx = depth_intrinsic[0,2]
    my = depth_intrinsic[1,2]
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth_img > 0)
    z = np.where(valid, depth_img / 1000.0, np.nan)
    x = np.where(valid, z * (c - mx) / fx, 0)
    y = np.where(valid, z * (r - my) / fy, 0)

    pts = np.dstack((x, y, z)).reshape(-1, 3)

    ptcloud = pts[~np.isnan(pts[:,2]), :]

    return ptcloud



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Check the reprojection from 3D to 2D images')
    parser.add_argument('--data_dir', type=str, default='/mnt/Data/Datasets/ScanNet_v2/scans/',
                        help='The path to annotations')
    parser.add_argument('--frame_skip', type=int, default=30,
                        help='the number of frames to skip in extracting instance annotation images')

    opt = parser.parse_args()

    # SCAN_NAMES = [line.rstrip() for line in open('/mnt/Data/Datasets/ScanNet_v1/sceneid_sort.txt')]
    SCAN_NAMES = ['scene0000_00']

    for scan_id, scan_name in enumerate(SCAN_NAMES):
        print('====== Process {0}-th scan [{1}] ======'.format(scan_id, scan_name))
        scan_path = os.path.join(opt.data_dir, scan_name)

        # parse the mesh file
        mesh_file = os.path.join(scan_path, '{0}_vh_clean_2.ply'.format(scan_name))
        mesh_vertices = scannet_utils.read_mesh_vertices_rgb(mesh_file)
        ptcloud = np.ones((mesh_vertices.shape[0], 4))
        ptcloud[:, 0:3] = mesh_vertices[:, 0:3]

        ### Load from <scene_id>.txt
        meta_file = os.path.join(scan_path, '{0}.txt'.format(scan_name))
        depth_intrinsic = scannet_utils.read_depth_intrinsic(meta_file)
        camera_intrinsic = scannet_utils.read_camera_intrinsic(meta_file)
        color2depth_extrinsic = scannet_utils.read_color2depth_extrinsic(meta_file)

        num_frames = len(os.listdir(os.path.join(scan_path, 'color')))
        for i in range(num_frames):
            frame_name = opt.frame_skip * i
            rgb_img_path = os.path.join(scan_path, 'color', '{0}.jpg'.format(frame_name))
            depth_img_path = os.path.join(scan_path, 'depth', '{0}.png'.format(frame_name))

            depth_img = np.array(Image.open(depth_img_path))
            pts_camera = depth_to_point_cloud(depth_img, depth_intrinsic)
            pts_camera_ext = np.ones((pts_camera.shape[0], 4))
            pts_camera_ext[:, 0:3] = pts_camera[:, 0:3]
            pts_camera_ext = np.dot(pts_camera_ext, color2depth_extrinsic.transpose())

            ## the matrix in /pose/<frameid>.txt is to map the camera coord to world coord
            camera_extrinsic_path = os.path.join(scan_path, 'pose', '{0}.txt'.format(frame_name))
            camera_extrinsic = np.loadtxt(camera_extrinsic_path)  # 4*4
            # transform one frame from camera coordinate to world coordinate
            pts_world = np.dot(pts_camera_ext, camera_extrinsic.transpose())

            # # transform whole scene from world coordinate to camera coordinate
            # ptcloud_camera = np.dot(ptcloud, np.linalg.inv(camera_extrinsic).transpose())  # N*4

            fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4), fgcolor=None, engine=None, size=(1000, 1000))
            mlab.points3d(ptcloud[:, 0], ptcloud[:, 1], ptcloud[:, 2], mode='point', color=(0,0,0), scale_factor=1, figure=fig)
            mlab.points3d(pts_world[:,0], pts_world[:,1], pts_world[:,2], color=(1, 1, 1), mode='point', scale_factor=1, figure=fig)
            mlab.orientation_axes()
            mlab.show()

            # transform from camera coordinate to pixel coordinate
            pts_pixel = np.dot(pts_camera_ext, camera_intrinsic.transpose()) #N*3
            pts_pixel[:, 0] /= pts_pixel[:, 2]
            pts_pixel[:, 1] /= pts_pixel[:, 2]
            valid_mask = (pts_pixel[:,0] >= 0) & (pts_pixel[:,0]<=1296) & (pts_pixel[:,1]>=0)& (pts_pixel[:,1]<=968) \
                         & (pts_pixel[:,2]>0)
            pts_image = pts_pixel[valid_mask, :]
            depth = pts_image[:,2]

            # visualize the reprojected points on color frame
            rgb_img = cv2.imread(rgb_img_path)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

            cmap = plt.cm.get_cmap('hsv', 256)
            cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

            for i in range(pts_image.shape[0]):
                d = depth[i]
                color = cmap[int(120.0 / d), :]
                cv2.circle(rgb_img, (int(np.round(pts_image[i, 0])), int(np.round(pts_image[i, 1]))), 2,
                           color=tuple(color), thickness=-1)
            Image.fromarray(rgb_img).show()