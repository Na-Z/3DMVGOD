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
from Box3dGenerator.numba_utils import *
from Box3dGenerator.bounding_box import *


def _calc_boundary_points(bbox2d_dimension, camera_intrinsic, z_near, z_far):
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
        print('\t Warning! The optimization is unsolved, the status of optimization result is {0}'.format(res.status))
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


def remove_noisy_frustums(frustum_planes, min_volume, thres_volume_ratio=2.):
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
            if cur_volume >= thres_volume_ratio*min_volume*i:
                to_remove_frustums.extend(to_exclude)
                break
        else:
            break
    # remove the noisy frustums
    if len(to_remove_frustums) != 0:
        to_remove_frustums = list(set(to_remove_frustums))
        print('\t Removed {0} noisy frustums [volume_ratio-{1}]'.format(len(to_remove_frustums), thres_volume_ratio))
        frustum_planes = [frustum_plane for idx, frustum_plane in enumerate(frustum_planes)
                                            if idx not in to_remove_frustums]
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


def frustum_ptcloud_with_cam_in_world_frame(depth_img, bbox2d_dimension, CAM, depth_intrinsic, color2depth_extrinsic,
                                             camera2world_extrinsic):
    pts_cam_depth_camera = utils.cropped_depth_to_point_cloud_with_cam(depth_img, depth_intrinsic, bbox2d_dimension, CAM)
    pts_depth_camera = pts_cam_depth_camera[:,:3]
    cam_score = pts_cam_depth_camera[:, -1].reshape(-1, 1)

    pts_color_camera = utils.calibrate_camera_depth_to_color(pts_depth_camera, color2depth_extrinsic)
    z_near = np.amin(pts_color_camera, axis=0)[2]
    z_far = np.amax(pts_color_camera, axis=0)[2]

    pts_world = pts_color_camera @ camera2world_extrinsic.transpose()

    pts_cam_world = np.hstack((pts_world[:, :3], cam_score))

    return pts_cam_world, z_near, z_far



def pcmerge(ptclouds, gridSize=.02):
    '''
    merge a set of 3D point clouds using a box grid filter
    :param: ptclouds, a stack of several frustum point cloud with shape (n, 4) [x,y,z,score]
                            where score indicate the class activated score
    :param: gridSize, size of the voxel for grid filter, specified as a numeric value.
                        Increase the size of gridSize when requiring a higher-resolution grid.
    :return: a merged point cloud
    '''

    def voxelize_pointcloud(n_xyz):
        segments = []
        for i in range(3):
            s = np.linspace(xyz_min[i], xyz_max[i], num=(n_xyz[i]))
            segments.append(s)

        ## find where each point lies in corresponding segmented axis
        voxel_x = np.clip(np.searchsorted(segments[0], ptclouds[:, 0]), 0, n_xyz[0]-1)
        voxel_y = np.clip(np.searchsorted(segments[1], ptclouds[:, 1]), 0, n_xyz[1]-1)
        voxel_z = np.clip(np.searchsorted(segments[2], ptclouds[:, 2]), 0, n_xyz[2]-1)
        voxel_n = np.ravel_multi_index([voxel_x, voxel_y, voxel_z], n_xyz)

        return voxel_n

    xyz_min = np.amin(ptclouds, axis=0)[0:3]
    xyz_max = np.amax(ptclouds, axis=0)[0:3]
    n_xyz = np.ceil((xyz_max-xyz_min)/gridSize).astype(np.int) #n_xyz = [nx, ny, nz]

    voxel_n = voxelize_pointcloud(n_xyz)
    n_voxels = n_xyz[0] * n_xyz[1] * n_xyz[2]

    voxel_sum = groupby_sum(ptclouds, voxel_n, np.zeros((n_voxels,4)))
    voxel_count = groupby_count(ptclouds, voxel_n, np.zeros(n_voxels))
    voxel_grid = np.nan_to_num(voxel_sum / voxel_count.reshape(-1,1))
    ptclouds_merged = voxel_grid[np.all(voxel_grid, axis=1)] #filter out empty voxels

    return ptclouds_merged


def compute_min_max_bounds_in_one_track(scan_dir, scan_name, objects, trajectory, cam_thres_ratio=.5, is_OBB=False):

    meta_file_path = os.path.join(scan_dir, '{0}.txt'.format(scan_name))
    axis_align_matrix, color2depth_extrinsic, camera_intrinsic, depth_intrinsic = utils.read_meta_file(meta_file_path)

    frustum_ptclouds = []
    frustum_planes = []
    instance_ids = []

    for frame_idx, bbox_idx in trajectory:
        obj = objects[frame_idx][bbox_idx]
        dimension = obj['dimension']
        classname = obj['classname']
        frame_name = obj['frame_name']
        instance_ids.append(obj['instance_id'])
        #visualize_bbox(scan_dir, obj, draw_text=False)

        depth_img_path = os.path.join(scan_dir, 'depth', '{0}.png'.format(frame_name))
        depth_img = np.array(Image.open(depth_img_path))

        #depth_img[depth_img == 0] = np.max(depth_img)
        #plt.imshow(1.0 / depth_img)
        #plt.show()

        camera2world_extrinsic_path = os.path.join(scan_dir, 'pose', '{0}.txt'.format(frame_name))
        camera2world_extrinsic = np.loadtxt(camera2world_extrinsic_path)  # 4*4

        ## generate frustum point cloud
        cam_path = os.path.join(scan_dir, 'cam', '{0}.npy'.format(frame_name))
        CAMs = np.load(cam_path)
        CAM = CAMs[:, :, cfg.SCANNET.CLASS2INDEX[classname]]
        frustum_ptcloud, z_near, z_far = frustum_ptcloud_with_cam_in_world_frame(depth_img, dimension, CAM,
                                            depth_intrinsic, color2depth_extrinsic, camera2world_extrinsic)
        frustum_ptclouds.append(frustum_ptcloud)

        ## generate frustum clipping planes
        frustum_plane = extract_frustum_plane(dimension, camera_intrinsic, camera2world_extrinsic, z_near, z_far)
        frustum_planes.append(frustum_plane)
        ## visualize single frustum
        # visualize_one_frustum(frustum_plane)
        # visualize_one_frustum_plus_points(frustum_plane, frustum_ptcloud)

    ## visualize the whole point cloud and the ground truth 3D bounding box
    instance_ids = np.unique(np.array(instance_ids))
    if len(instance_ids) == 1:
        visualize_bbox3d_in_whole_scene(scan_dir, scan_name, axis_align_matrix, instance_ids[0])
    else:
        ## debug to check if one track contains more than one instance..
        print('Warning! The track contains more than one object!')
        for instance_id in instance_ids:
            visualize_bbox3d_in_whole_scene(scan_dir, scan_name, axis_align_matrix, instance_id)

    ptclouds_multiview = np.vstack(frustum_ptclouds)
    ## merge the frustum point clouds from multiple views
    ptclouds_merged = pcmerge(ptclouds_multiview)
    visualize_frustum_ptcloud_with_cam(ptclouds_merged)
    visualize_n_frustums_plus_ptclouds(frustum_planes, ptclouds_multiview)

    ## compute the intersection of n frustums, and remove noisy frustums
    hs = frustum_planes_intersect(frustum_planes, visu_intersection_points=False)

    if hs is not None:
        min_volume = ConvexHull(hs.intersections).volume
        frustum_planes_clean = remove_noisy_frustums(frustum_planes, min_volume, thres_volume_ratio=1.5)
        # visualize_n_frustums_plus_ptclouds(frustum_planes_clean, ptclouds)
        hs_clean = frustum_planes_intersect(frustum_planes_clean, visu_intersection_points=False)
        intersection_points = hs_clean.intersections
        inside_mask = in_hull(ptclouds_merged[:,:3],intersection_points)
        inside_ptcloud = ptclouds_merged[inside_mask]

        visualize_convex_hull_plus_ptcloud_interactive(intersection_points, ptclouds_merged[:,:3], inside_mask)

        visualize_frustum_ptcloud_with_cam(inside_ptcloud) #debug

        ## select relatively high class activated points
        avg_cam_score = np.mean(inside_ptcloud, axis=0)[3]
        activate_mask = np.where(inside_ptcloud[:,3]>cam_thres_ratio*avg_cam_score)[0]
        select_ptcloud = inside_ptcloud[activate_mask]
        visualize_frustum_ptcloud_with_cam(select_ptcloud) # debug

        candidate_pts = utils.align_world_with_axis(select_ptcloud[:,:3], axis_align_matrix)

        if is_OBB:
            #TODO: oriented bounding box
            pass
        else:
            ## AABB (axis-aligned bounding box) is generated
            bbox3d = create_AABB(candidate_pts, cfg.SCANNET.CLASS2INDEX[classname])
            visualize_bbox3d_in_whole_scene(scan_dir, scan_name, axis_align_matrix, instance_ids[0], bbox3d)
        return bbox3d

    else:
        return None









