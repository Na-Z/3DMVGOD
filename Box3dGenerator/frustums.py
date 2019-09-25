import os
import numpy as np
from PIL import Image
import copy
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection
from scipy.spatial import ConvexHull

from config import cfg
import data.scannet.scannet_utils as utils
from Box3dGenerator.visualizer import *


def _calc_boundary_points(bbox2d_dimension, camera_intrinsic, n=-0.1, f=-5):
    '''
    calculate the six (right, left, bottom, top, near, far) boundary values of the truncated frustum
     along x, y, z axis based on the 2D bbox corner points of color image
    '''
    xmin, ymin, xmax, ymax = bbox2d_dimension

    #left bottom point of the bbox on color image
    pts_lb_image = np.ones((1, 3))
    pts_lb_image[0, 0] = xmin
    pts_lb_image[0, 1] = ymin
    pts_lb_image[0, 2] = -n

    # right top point of the bbox on color image
    pts_rt_image = np.ones((1, 3))
    pts_rt_image[0, 0] = xmax
    pts_rt_image[0, 1] = ymax
    pts_rt_image[0, 2] = -n

    # transform from the color image coord to color camera coord
    pts_lb_camera = utils.project_image_to_camera(pts_lb_image, camera_intrinsic) #(1,3)
    pts_rt_camera = utils.project_image_to_camera(pts_rt_image, camera_intrinsic)

    l = pts_lb_camera[0, 0]
    b = pts_lb_camera[0, 1]
    r = pts_rt_camera[0, 0]
    t = pts_rt_camera[0, 1]

    return r, l, b, t, n, f



def _construct_projection_matrix(r, l, b, t, n, f):
    '''
    projection matrix
    P =
    | 2n/(r-l)    0      (r+l)/(r-l)      0     |
    |     0     2n/(t-b) (t+b)/(t-b)      0     |
    |     0       0     -(f+n)/(f-n) -2fn/(f-n) |
    |     0       0           -1          0     |
    :return:
    '''
    P = np.zeros((4,4))
    P[0,0] = 2*n / (r-l)
    P[0,2] = (r+l) / (r-l)
    P[1,1] = 2*n / (t-b)
    P[1,2] = (t+b) / (t-b)
    P[2,2] = -(f+n) / (f-n)
    P[2,3] = -2*f*n / (f-n)
    P[3,2] = -1

    return P


def _normalize_plane(p_planes):
    n = p_planes.shape[0]
    for i in range(n):
        mag = np.sqrt(p_planes[i,0]**2 + p_planes[i,1]**2 + p_planes[i,2]**2)
        p_planes[i, :] /= mag

    return p_planes


def _calc_inequalities_coefficients(M):
    '''
    calculate inequalities coefficients for clipping plane of one frustum
    :param M: np.ndarray, shape (4,4), projection matrix
    :return: p_planes: np.ndarray, shape (6,4), coefficients for six clipping plane equations
                        The order of the rows is left-right-bottom-top-near-far.
                        Each row contains four elements representing coefficients (a,b,c,d)
                        for inequality (ax+by+cz+d>0)
    '''
    normals = np.zeros((6,4))
    normals[:, 3] = 1
    for i in range(normals.shape[0]):
        normals[i, i//2] = 1 - (i%2)*2
    p_planes = normals @ M

    # p_planes = np.zeros((6,4))
    #
    # # left clipping plane
    # p_planes[0, 0] = M[3, 0] + M[0, 0]
    # p_planes[0, 1] = M[3, 1] + M[0, 1]
    # p_planes[0, 2] = M[3, 2] + M[0, 2]
    # p_planes[0, 3] = M[3, 3] + M[0, 3]
    #
    # # right clipping plane
    # p_planes[1, 0] = M[3, 0] - M[0, 0]
    # p_planes[1, 1] = M[3, 1] - M[0, 1]
    # p_planes[1, 2] = M[2, 3] - M[0, 2]
    # p_planes[1, 3] = M[3, 3] - M[0, 3]
    #
    # # bottom clipping plane
    # p_planes[2, 0] = M[3, 0] + M[1, 0]
    # p_planes[2, 1] = M[3, 1] + M[1, 1]
    # p_planes[2, 2] = M[3, 2] + M[1, 2]
    # p_planes[2, 3] = M[3, 3] + M[1, 3]
    #
    # # top clipping plane
    # p_planes[3, 0] = M[3, 0] - M[1, 0]
    # p_planes[3, 1] = M[3, 1] - M[1, 1]
    # p_planes[3, 2] = M[3, 2] - M[1, 2]
    # p_planes[3, 3] = M[3, 3] - M[1, 3]
    #
    # # near clipping plane
    # p_planes[4, 0] = M[3, 0] + M[2, 0]
    # p_planes[4, 1] = M[3, 1] + M[2, 1]
    # p_planes[4, 2] = M[3, 2] + M[2, 2]
    # p_planes[4, 3] = M[3, 3] + M[2, 3]
    #
    # # far clipping plane
    # p_planes[5, 0] = M[3, 0] - M[2, 0]
    # p_planes[5, 1] = M[3, 1] - M[2, 1]
    # p_planes[5, 2] = M[3, 2] - M[2, 2]
    # p_planes[5, 3] = M[3, 3] - M[2, 3]

    p_planes = _normalize_plane(p_planes)

    return p_planes


def _calc_interior_point(halfspaces):
    c = np.zeros((halfspaces.shape[1]-1,))
    A = halfspaces[:, :-1]
    b = -halfspaces[:, -1]
    res = linprog(c, A_ub=A, b_ub=b, bounds=(0,None))
    # if the problem can not be solved, return None
    if res.status != 0:
        print('Warning! The optimization is unsolved, the status of optimization result is {0}'.format(res.status))
        return None
    interior_point = res.x
    return interior_point


def frustum_planes_intersect(p_planes_list):
    halfspaces = np.vstack(p_planes_list)
    ## change halfspaces from stacked Inequalities of the form Ax+b>0 in format [A; b] to -Ax-b<0 in format [-A;-b]
    #halfspaces = - halfspaces
    interior_point = _calc_interior_point(halfspaces)
    if interior_point is None:
        return None
    visualize_frustums_plus_interior_point(p_planes_list, interior_point)
    hs = HalfspaceIntersection(halfspaces, interior_point)
    visualize_frustums_intersection(p_planes_list, hs.intersections)
    return hs


def extract_frustum_plane(bbox2d_dimension, camera_intrinsic, camera2world_extrinsic):

    r, l, b, t, n, f = _calc_boundary_points(bbox2d_dimension, camera_intrinsic)
    P = _construct_projection_matrix(r, l, b, t, n, f)
    M = P @ np.linalg.inv(camera2world_extrinsic)
    p_planes = _calc_inequalities_coefficients(M)

    return p_planes


def frustum_ptcloud_with_cam_in_world_frame(depth_img, bbox2d_dimension, CAM, depth_intrinsic,
                                            color2depth_extrinsic, camera2world_extrinsic):
    pts_cam_depth_camera = utils.cropped_depth_to_point_cloud_with_cam(depth_img, depth_intrinsic, bbox2d_dimension, CAM)
    pts_depth_camera = pts_cam_depth_camera[:,:3]
    cam_score = pts_cam_depth_camera[:, -1].reshape(-1, 1)

    pts_color_camera = utils.calibrate_camera_depth_to_color(pts_depth_camera, color2depth_extrinsic)
    pts_world = pts_color_camera @ camera2world_extrinsic.transpose()

    pts_cam_world = np.hstack((pts_world[:, :3], cam_score))
    return pts_cam_world


def compute_min_max_bounds_in_one_track(scan_dir, objects, trajectory):

    frustum_ptclouds = []
    frustum_planes = []

    for frame_idx, bbox_idx in trajectory:
        obj = objects[frame_idx][bbox_idx]
        dimension = obj['dimension']
        classname = obj['classname']
        frame_name = obj['frame_name']
        # visualize_bbox(scan_dir, obj)

        depth_img_path = os.path.join(scan_dir, 'depth', '{0}.png'.format(frame_name))
        depth_img = np.array(Image.open(depth_img_path))

        camera2world_extrinsic_path = os.path.join(scan_dir, 'pose', '{0}.txt'.format(frame_name))
        camera2world_extrinsic = np.loadtxt(camera2world_extrinsic_path)  # 4*4

        meta_file_path = os.path.join(scan_dir, '{0}.txt'.format(scan_dir.split('/')[-1]))
        depth_intrinsic = utils.read_depth_intrinsic(meta_file_path)
        camera_intrinsic = utils.read_camera_intrinsic(meta_file_path)
        color2depth_extrinsic = utils.read_color2depth_extrinsic(meta_file_path)

        ## generate frustum point cloud
        cam_path = os.path.join(scan_dir, 'cam', '{0}.npy'.format(frame_name))
        CAMs = np.load(cam_path)
        CAM = CAMs[:, :, cfg.SCANNET.CLASS2INDEX[classname]]
        frustum_ptcloud = frustum_ptcloud_with_cam_in_world_frame(depth_img, dimension, CAM, depth_intrinsic,
                                                color2depth_extrinsic, camera2world_extrinsic)
        ## visualize frustum point cloud
        #visualize_frustum_ptcloud_with_cam(frustum_ptcloud)
        frustum_ptclouds.append(frustum_ptcloud)

        ## generate frustum clipping planes
        frustum_plane = extract_frustum_plane(dimension, camera_intrinsic, camera2world_extrinsic)
        ## visualize frustum plan
        # visualize_one_frustum(frustum_plane)
        # visualize_one_frustum_plus_points(frustum_plane, frustum_ptcloud)
        frustum_planes.append(frustum_plane)

    #TODO: use activated frustum ptcloud to filter out invalid object..

    visualize_n_frustums(frustum_planes[:3])
    hs = frustum_planes_intersect([frustum_planes[0], frustum_planes[1]])
    if hs is not None:
        print(hs.dual_vertices)

    # plot the convex hull






