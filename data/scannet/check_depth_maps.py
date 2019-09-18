import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from mayavi import mlab

import scannet_utils
from config import cfg


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Check the reprojection from 3D to 2D images')
    parser.add_argument('--data_dir', type=str, default='/mnt/Data/Datasets/ScanNet_v2/scans/',
                        help='The path to annotations')

    opt = parser.parse_args()

    SCAN_NAMES = [line.rstrip() for line in open('/mnt/Data/Datasets/ScanNet_v1/sceneid_sort.txt')]

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

            frame_name = cfg.SCANNET.FRAME_SKIP * i
            rgb_img_path = os.path.join(scan_path, 'color', '{0}.jpg'.format(frame_name))
            depth_img_path = os.path.join(scan_path, 'depth', '{0}.png'.format(frame_name))

            depth_img = np.array(Image.open(depth_img_path))
            pts_camera = scannet_utils.depth_to_point_cloud(depth_img, depth_intrinsic)
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