'''Check the reprojection from 3D to 2D images
Author: Zhao Na
Date: Sep 2019
'''
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


if __name__ == '__main__':

    DATA_DIR = '/mnt/Data/Datasets/ScanNet_v2/scans/'

    # SCAN_NAMES = [line.rstrip() for line in open('/mnt/Data/Datasets/ScanNet_v1/sceneid_sort.txt')]
    SCAN_NAMES = ['scene0000_00']

    for scan_id, scan_name in enumerate(SCAN_NAMES):
        print('====== Process {0}-th scan [{1}] ======'.format(scan_id, scan_name))
        scan_path = os.path.join(DATA_DIR, scan_name)

        # parse the camera intrinsic file
        ## WHY the parameters from intrinsic_color.txt are different from those in scene_id.txt file??
        ## The one in scene_id.txt should be after camera calibration (undistortion)

        # ### Load from intrinsic_color.txt
        # camera_intrinsic_path = os.path.join(scan_path, 'intrinsic', 'intrinsic_color.txt')
        # camera_intrinsic = np.loadtxt(camera_intrinsic_path)[:3,:]

        ### Load from scene_id.txt
        meta_file = os.path.join(scan_path, '{0}.txt'.format(scan_name))
        camera_intrinsic = scannet_utils.read_camera_intrinsic(meta_file)

        # parse the mesh file
        mesh_file = os.path.join(scan_path, '{0}_vh_clean_2.ply'.format(scan_name))
        mesh_vertices = scannet_utils.read_mesh_vertices_rgb(mesh_file)

        ## visualize-1
        # plotxyzrgb(mesh_vertices, block=True)

        ## visualize-2
        fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 1000))
        mlab.points3d(mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_vertices[:, 2], mesh_vertices[:, 2], mode='point',
                      colormap='gnuplot', figure=fig)
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.3)
        mlab.orientation_axes()
        mlab.show()

        pts = np.ones((mesh_vertices.shape[0], 4))
        pts[:, 0:3] = mesh_vertices[:, 0:3]

        framenames = os.listdir(os.path.join(scan_path, 'color'))
        for frame_idx in framenames:
            instance_img_path = os.path.join(scan_path, 'instance-filt', '{0}.png'.format(frame_idx))

            ## the matrix in /pose/<frameid>.txt is to map the camera coord to world coord
            camera_extrinsic_path = os.path.join(scan_path, 'pose', '{0}.txt'.format(frame_idx))
            camera_extrinsic =  np.loadtxt(camera_extrinsic_path) #4*4
            # transform from world coordinate to camera coordinate
            pts_camera = np.dot(pts, np.linalg.inv(camera_extrinsic).transpose()) #N*4

            ## visualize-1
            vtkControl = get_vtk_control(True)
            plotxyzrgb(np.hstack((pts_camera[:,:3], mesh_vertices[:,3:6])))
            vtkControl.AddAxesActor(1.0)
            vtkControl.exec_()

            ## visualize-2
            # fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 1000))
            # mlab.points3d(pts_camera[:, 0], pts_camera[:, 1], pts_camera[:, 2], pts_camera[:, 2], mode='point',
            #               colormap='gnuplot', figure=fig)
            # mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.3)
            # mlab.orientation_axes()
            # mlab.show()

            # transform from camera coordinate to pixel coordinate
            pts_pixel = np.dot(pts_camera, camera_intrinsic.transpose()) #N*3
            pts_pixel[:, 0] /= pts_pixel[:, 2]
            pts_pixel[:, 1] /= pts_pixel[:, 2]
            valid_mask = (pts_pixel[:,0] >= 0) & (pts_pixel[:,0]<=1296) & (pts_pixel[:,1]>=0)& (pts_pixel[:,1]<=968) \
                         & (pts_pixel[:,2]>0)
            pts_image = pts_pixel[valid_mask, :]
            depth = pts_image[:,2]
            plotxyzrgb(np.hstack((pts_image, mesh_vertices[valid_mask, 3:6])), block=True)

            # visualize the reprojected points on color frame
            rgb_img = instance_img_path.replace('instance-filt', 'color').replace('png', 'jpg')
            rgb_img = cv2.imread(rgb_img)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

            cmap = plt.cm.get_cmap('hsv', 256)
            cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

            for i in range(pts_image.shape[0]):
                d = depth[i]
                color = cmap[int(120.0 / d), :]
                cv2.circle(rgb_img, (int(np.round(pts_image[i, 0])), int(np.round(pts_image[i, 1]))), 2,
                           color=tuple(color), thickness=-1)
            Image.fromarray(rgb_img).show()