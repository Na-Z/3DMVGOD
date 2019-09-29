import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, ConvexHull
import itertools

from config import cfg
import data.scannet.scannet_utils as utils
from Box3dGenerator.visualizer import *


def _calc_boundary_points(bbox2d_dimension, camera_intrinsic, z_near=0.1, z_far=5):
    '''
    calculate the six (right, left, bottom, top, near, far) boundary values of the truncated frustum
     along x, y, z axis based on the 2D bbox corner points of color image
    '''
    # the camera at the origin is looking along -Z axis in eye space, we need to negate the input positive values
    # refer to http://www.songho.ca/opengl/gl_projectionmatrix.html
    n = -z_near
    f = -z_far

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


def extract_frustum_plane(bbox2d_dimension, camera_intrinsic, camera2world_extrinsic, z_near, z_far):

    r, l, b, t, n, f = _calc_boundary_points(bbox2d_dimension, camera_intrinsic, z_near, z_far)
    P = _construct_projection_matrix(r, l, b, t, n, f)
    M = P @ np.linalg.inv(camera2world_extrinsic)
    p_planes = _calc_inequalities_coefficients(M)

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


def frustum_planes_intersect(p_planes_list, visu_interior_point=False, visu_intersection_points=False):
    halfspaces = np.vstack(p_planes_list)
    ## change halfspaces from stacked Inequalities of the form Ax+b>0 in format [A; b] to -Ax-b<0 in format [-A;-b]
    #halfspaces = - halfspaces
    interior_point = _calc_interior_point(halfspaces)
    if interior_point is None:
        return None
    if visu_interior_point:
        visualize_frustums_plus_interior_point(p_planes_list, interior_point)
    hs = HalfspaceIntersection(halfspaces, interior_point)
    if visu_intersection_points:
        visualize_frustums_intersection(p_planes_list, hs.intersections)
    return hs


def remove_noisy_frustums(frustum_planes, min_volume, thres_ratio=2.):
    '''compute the intersection of (n-i) frustums, if the intersection volume is larger than thres_ratio*min_volume,
        consider those i frustums as noisy frustums...
    '''
    n = len(frustum_planes) # number of frustums
    iterable = list(range(n))
    to_remove_frustums = []
    for i in range(1, (n//2)+1):
        exclude_pool = itertools.combinations(iterable, i)
        for to_exclude in exclude_pool:
            cur_frustum_planes = [frustum_plane for idx, frustum_plane in enumerate(frustum_planes)
                                                    if idx not in to_exclude]
            cur_hs = frustum_planes_intersect(cur_frustum_planes, visu_intersection_points=False)
            cur_volume = ConvexHull(cur_hs.intersections).volume
            # TODO: if increase the threhold i times when considering remove i (i>1) frustums
            if cur_volume >= thres_ratio*min_volume*i:
                to_remove_frustums.extend(to_exclude)
                break
        else:
            break
    # remove the noisy frustums
    if len(to_remove_frustums) != 0:
        frustum_planes = [frustum_plane for idx, frustum_plane in enumerate(frustum_planes)
                                            if idx not in list(set(to_remove_frustums))]
    return frustum_planes


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0


def frustum_ptcloud_with_cam_in_world_frame(depth_img, bbox2d_dimension, CAM, depth_intrinsic,
                                            color2depth_extrinsic, camera2world_extrinsic):
    pts_cam_depth_camera = utils.cropped_depth_to_point_cloud_with_cam(depth_img, depth_intrinsic, bbox2d_dimension, CAM)
    pts_depth_camera = pts_cam_depth_camera[:,:3]
    cam_score = pts_cam_depth_camera[:, -1].reshape(-1, 1)

    pts_color_camera = utils.calibrate_camera_depth_to_color(pts_depth_camera, color2depth_extrinsic)
    z_near = np.amin(pts_color_camera, axis=0)[2]
    z_far = np.amax(pts_color_camera, axis=0)[2]

    pts_world = pts_color_camera @ camera2world_extrinsic.transpose()

    pts_cam_world = np.hstack((pts_world[:, :3], cam_score))
    return pts_cam_world, z_near, z_far


def compute_min_max_bounds_in_one_track(scan_dir, scan_name, objects, trajectory):

    frustum_ptclouds = []
    frustum_planes = []
    instance_ids = []

    for frame_idx, bbox_idx in trajectory:
        obj = objects[frame_idx][bbox_idx]
        dimension = obj['dimension']
        classname = obj['classname']
        frame_name = obj['frame_name']
        instance_ids.append(obj['instance_id'])
        # visualize_bbox(scan_dir, obj)

        depth_img_path = os.path.join(scan_dir, 'depth', '{0}.png'.format(frame_name))
        depth_img = np.array(Image.open(depth_img_path))

        camera2world_extrinsic_path = os.path.join(scan_dir, 'pose', '{0}.txt'.format(frame_name))
        camera2world_extrinsic = np.loadtxt(camera2world_extrinsic_path)  # 4*4

        meta_file_path = os.path.join(scan_dir, '{0}.txt'.format(scan_name))
        depth_intrinsic = utils.read_depth_intrinsic(meta_file_path)
        camera_intrinsic = utils.read_camera_intrinsic(meta_file_path)
        color2depth_extrinsic = utils.read_color2depth_extrinsic(meta_file_path)

        ## generate frustum point cloud
        cam_path = os.path.join(scan_dir, 'cam', '{0}.npy'.format(frame_name))
        CAMs = np.load(cam_path)
        CAM = CAMs[:, :, cfg.SCANNET.CLASS2INDEX[classname]]
        frustum_ptcloud, z_near, z_far = frustum_ptcloud_with_cam_in_world_frame(depth_img, dimension, CAM,
                                            depth_intrinsic, color2depth_extrinsic, camera2world_extrinsic)
        ## visualize frustum point cloud
        #visualize_frustum_ptcloud_with_cam(frustum_ptcloud)
        frustum_ptclouds.append(frustum_ptcloud)

        ## generate frustum clipping planes
        frustum_plane = extract_frustum_plane(dimension, camera_intrinsic, camera2world_extrinsic, z_near, z_far)
        ## visualize frustum plan
        # visualize_one_frustum(frustum_plane)
        # visualize_one_frustum_plus_points(frustum_plane, frustum_ptcloud)
        frustum_planes.append(frustum_plane)


    ## visualize the whole point cloud and the ground truth 3D bounding box
    instance_ids = np.unique(np.array(instance_ids))
    if len(instance_ids) == 1:
        visualize_bbox3d_in_whole_scene(scan_dir, scan_name, instance_ids[0])
    else:
        ## debug to check if one track contains more than one instance..
        print('Warning! The track contains more than one object!')
        for instance_id in instance_ids:
            visualize_bbox3d_in_whole_scene(scan_dir, scan_name, instance_id)

    # visualize_n_frustums(frustum_planes)
    ptclouds = np.vstack(frustum_ptclouds)
    visualize_n_frustums_plus_ptclouds(frustum_planes, ptclouds)

    ## compute the intersection of n frustums
    hs = frustum_planes_intersect(frustum_planes, visu_intersection_points=True)
    if hs is not None:
        min_volume = ConvexHull(hs.intersections).volume

        frustum_planes_clean = remove_noisy_frustums(frustum_planes, min_volume, thres_ratio=1.5)
        visualize_n_frustums_plus_ptclouds(frustum_planes_clean, ptclouds)
        hs = frustum_planes_intersect(frustum_planes_clean, visu_intersection_points=False)
        intersection_points = hs.intersections
        inside_mask = in_hull(ptclouds[:,:3],intersection_points)

        # visualize_convex_hull_plus_ptcloud_static(intersection_points, ptclouds[:,:3], inside_mask)
        visualize_convex_hull_plus_ptcloud_interactive(intersection_points, ptclouds[:,:3], inside_mask)








