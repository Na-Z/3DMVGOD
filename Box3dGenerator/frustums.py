import os
import numpy as np
from PIL import Image
import cv2

from config import cfg
import data.scannet.scannet_utils as utils
from Box3dGenerator.visualizer import *


def _calc_boundary_points(depth_img, depth_intrinsic, color2depth_extrinsic, camera2world_extrinsic, bbox2d_dimension,
                          near_thr=0.1, far_thr=5):
    '''
    calculate the six (right, left, bottom, top, near, far) boundary points
    of the truncated frustum based on the 2D bbox corner points of color image
    '''
    xmin, ymin, xmax, ymax = utils._rescale_bbox_dim(bbox2d_dimension)

    #left bottom point of the bbox on depth image
    pts_lb_image = np.ones((1, 3))
    pts_lb_image[0, 0] = xmin
    pts_lb_image[0, 1] = ymin
    pts_lb_image[0, 2] = depth_img[ymin, xmin]

    # right top point of the bbox on depth image
    pts_rt_image = np.ones((1, 3))
    pts_rt_image[0, 0] = xmax
    pts_rt_image[0, 1] = ymax
    pts_rt_image[0, 2] = depth_img[ymax, xmax]

    # transform from the depth image coord to depth camera coord
    pts_lb_depth_camera = utils.project_image_to_camera(pts_lb_image, depth_intrinsic) #(1,3)
    pts_rt_depth_camera = utils.project_image_to_camera(pts_rt_image, depth_intrinsic)

    # transfrom from the depth camera coord to color camera coord
    pts_lb_color_camera = utils.calibrate_camera_depth_to_color(pts_lb_depth_camera, color2depth_extrinsic) #(1,4)
    pts_rt_color_camera = utils.calibrate_camera_depth_to_color(pts_rt_depth_camera, color2depth_extrinsic)

    # transform from the color camera coord to the world coord
    pts_lb_world = pts_lb_color_camera @ camera2world_extrinsic
    pts_rt_world = pts_rt_color_camera @ camera2world_extrinsic

    l = pts_lb_world[0, 0]
    b = pts_lb_world[0, 1]
    r = pts_rt_world[0, 0]
    t = pts_rt_world[0, 1]
    n = min(pts_lb_world[0, 2], pts_rt_world[0, 2])
    f = max(pts_lb_world[0, 2], pts_rt_world[0, 2])

    # check the n and f with the pre-defined thresholds, choose the proper one
    #TODO: use original z_near and z_far to compute frustums intersection
    n = min(n, near_thr)
    f = max(f, far_thr)

    return r, l, b, t, f, n


def _construct_projection_matrix(r, l, b, t, f, n):
    '''
    projection matrix
    M =
    | 2n/(r-l)    0      (r+l)/(r-l)      0     |
    |     0     2n/(t-b) (t+b)/(t-b)      0     |
    |     0       0     -(f+n)/(f-n) -2fn/(f-n) |
    |     0       0           -1          0     |
    :return:
    '''
    M = np.zeros((4,4))
    M[0,0] = 2*n / (r-l)
    M[0,2] = (r+l) / (r-l)
    M[1,1] = 2*n / (t-b)
    M[1,2] = (t+b) / (t-b)
    M[2,2] = -(f+n) / (f-n)
    M[2,3] = -2*f*n / (f-n)
    M[3,2] = -1

    return M


def _calc_inequalities_coefficients(M):
    '''
    calculate inequalities coefficients for clipping plane of one frustum
    :param M: np.ndarray, shape (4,4), projection matrix
    :return: p_planes: np.ndarray, shape (6,4), coefficients for six clipping plane equations
                        The order of the rows is left-right-bottom-top-near-far.
                        Each row contains four elements representing coefficients (a,b,c,d)
                        for inequality (ax+by+cz+d>0)
    '''
    p_planes = np.zeros((6,4))

    # left clipping plane
    p_planes[0, 0] = M[0, 3] + M[0, 0]
    p_planes[0, 1] = M[1, 3] + M[1, 0]
    p_planes[0, 2] = M[2, 3] + M[2, 0]
    p_planes[0, 3] = M[3, 3] + M[3, 0]

    # right clipping plane
    p_planes[1, 0] = M[0, 3] - M[0, 0]
    p_planes[1, 1] = M[1, 3] - M[1, 0]
    p_planes[1, 2] = M[2, 3] - M[2, 0]
    p_planes[1, 3] = M[3, 3] - M[3, 0]

    # bottom clipping plane
    p_planes[2, 0] = M[0, 3] + M[0, 1]
    p_planes[2, 1] = M[1, 3] + M[1, 1]
    p_planes[2, 2] = M[2, 3] + M[2, 1]
    p_planes[2, 3] = M[3, 3] + M[3, 1]

    # top clipping plane
    p_planes[3, 0] = M[0, 3] - M[0, 1]
    p_planes[3, 1] = M[1, 3] - M[1, 1]
    p_planes[3, 2] = M[2, 3] - M[2, 1]
    p_planes[3, 3] = M[3, 3] - M[3, 1]

    # near clipping plane
    p_planes[4, 0] = M[0, 3] + M[0, 2]
    p_planes[4, 1] = M[1, 3] + M[1, 2]
    p_planes[4, 2] = M[2, 3] + M[2, 2]
    p_planes[4, 3] = M[3, 3] + M[3, 2]

    # far clipping plane
    p_planes[5, 0] = M[0, 3] - M[0, 2]
    p_planes[5, 1] = M[1, 3] - M[1, 2]
    p_planes[5, 2] = M[2, 3] - M[2, 2]
    p_planes[5, 3] = M[3, 3] - M[3, 2]

    return p_planes


def frustum_planes_intersect():
    #TODO: compute the intweior point and implement scipy halfplane intersection
    pass


def extract_frustum_plane(bbox2d_dimension, depth_img, depth_intrinsic, camera2world_extrinsic, color2depth_extrinsic):

    r, l, b, t, f, n = _calc_boundary_points(depth_img, depth_intrinsic, color2depth_extrinsic,
                                             camera2world_extrinsic, bbox2d_dimension)
    M = _construct_projection_matrix(r, l, b, t, f, n)
    p_planes = _calc_inequalities_coefficients(M)

    return p_planes


def compute_min_max_bounds_in_one_track(scan_dir, objects, trajectory):

    frustum_ptclouds = []
    frustum_planes = []

    for frame_idx, bbox_idx in trajectory:
        obj = objects[frame_idx][bbox_idx]
        dimension = obj['dimension']
        classname = obj['classname']
        frame_name = obj['frame_name']

        depth_img_path = os.path.join(scan_dir, 'depth', '{0}.png'.format(frame_name))
        depth_img = np.array(Image.open(depth_img_path))

        camera2world_extrinsic_path = os.path.join(scan_dir, 'pose', '{0}.txt'.format(frame_name))
        camera2world_extrinsic = np.loadtxt(camera2world_extrinsic_path).transpose()  # 4*4

        meta_file_path = os.path.join(scan_dir, '{0}.txt'.format(scan_dir.split('/')[-1]))
        depth_intrinsic = utils.read_depth_intrinsic(meta_file_path)
        color2depth_extrinsic = utils.read_color2depth_extrinsic(meta_file_path)

        # generate frustum point cloud
        cam_path = os.path.join(scan_dir, 'cam', '{0}.npy'.format(frame_name))
        CAMs = np.load(cam_path)
        CAM = CAMs[:, :, cfg.SCANNET.CLASS2INDEX[classname]]

        frustum_ptcloud = utils.cropped_depth_to_point_cloud_with_cam(depth_img, depth_intrinsic, dimension, CAM)
        # visualize frustum point cloud
        visualize_bbox(scan_dir, obj)
        visualize_frustum_ptcloud_with_cam(frustum_ptcloud)
        frustum_ptclouds.append(frustum_ptcloud)

        # generate frustum clipping planes
        frustum_plane = extract_frustum_plane(dimension, depth_img, depth_intrinsic, camera2world_extrinsic,
                                              color2depth_extrinsic)
        frustum_planes.append(frustum_plane)

    #TODO: use activated frustum ptcloud to filter out invalid object..

    frustum_planes = np.vstack(frustum_planes)


