#!/usr/bin/python3
import sys
sys.path.append('../')
import os
import math
import random
import numpy as np
from timeit import default_timer as timer
from PIL import Image
import copy
import itertools
from scipy.spatial.transform import Rotation as Rot

from config import cfg
import data.scannet.scannet_utils as utils
from Box3dGenerator.visualizer import *


def _calculate_relative_angle(rot_a, rot_b):
    '''
    calculate the relative angle between two frames
    :param rot_a: rotation matrix of frame a, shape (3,3)
    :param rot_b: rotation matrix of frame b, shape (3,3)
    :return: the relative angle (degree) between frame a and b
    '''
    R_relative = (np.linalg.inv(rot_b) @ rot_a)
    r = Rot.from_dcm(R_relative)
    # quat = r.as_quat()
    # euler = r.as_euler('zyx')
    ## compute the euler vector
    ## (refer to https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Rotation_vector)
    rot_vec = r.as_rotvec()
    angle = np.degrees(np.linalg.norm(rot_vec))
    # print('rot_vec: {0}; \tangle:{1}'.format(rot_vec, angle))
    return angle


# def essential_matrix(poses, src_frame_idx, dst_frame_idx):
#     relative_pose = np.dot(np.linalg.inv(poses[dst_frame_idx]), poses[src_frame_idx])  # (4,4)
#     R = relative_pose[:3, :3]  # (3*3)
#     t = relative_pose[:3, 3]  # (3,)
#     t_x = np.array([[0, -t[2], t[1]],
#                     [t[2], 0, -t[0]],
#                     [-t[1], t[0], 0]])
#     E = np.dot(t_x, R)
#     return E
#
#
# def algebraic_distance(poses, K, src_frame_idx, dst_frame_idx, p_S, p_D):
#     '''
#     --> refer to Section 4.2 in 'Fundamental Matrix Estimation:  A Study of Error Criteria'
#         algebraic distance R = R(x_s, x_d) = (x_d)^T * F * (x_s)
#     :param p_S: point in image I_S, numpy array, (3,1)
#     :param p_D: point(s) in image I_D, numpy array, (3,n)
#              I_S and I_D are two perspective images of the same scene
#     :return: R: algebraic epipolar distance:  scalar
#              l_D: epipolar line in I_D that corresponds to x_s
#              l_S: epipolar line in I_S that corresponds to x_d
#     '''
#     E = essential_matrix(poses, src_frame_idx, dst_frame_idx)  # (3,3)
#     F = np.linalg.inv(K.transpose()).dot(E).dot(np.linalg.inv(K)) #fundamental matrix
#     l_D = np.dot(F, p_S)  # epipolar line (3,1)
#     l_S = np.dot(F.transpose(), p_D) #epipolar lines (3, n)
#     R = np.dot(p_D.transpose(), l_D).reshape(-1)  # (n)
#     return R, l_D, l_S
#
#
# def symmetric_epipolar_distance(poses, K, src_frame_idx, dst_frame_idx, p_S, p_D):
#     '''
#     --> refer to Section 4.3 in 'Fundamental Matrix Estimation:  A Study of Error Criteria'
#         It measures the geometric distance of each point to it sepipolar line
#
#     :param p_S: point in image I_S, numpy array, (3,1)
#     :param p_D: point(s) in image I_D, numpy array, (3,n)
#              I_S and I_D are two perspective images of the same scene
#     :return: SED_square: symmtric epipolar distance: numpy array, each element is a non-negative scalar
#              l: epipolar line
#     '''
#     R, l_D, l_S = algebraic_distance(poses, K, src_frame_idx, dst_frame_idx, p_S, p_D)
#     n_pts = p_D.shape[1]
#     SED_square = np.zeros((n_pts))
#     for i in range(n_pts):
#         SED_square[i] = (1/(l_S[0,i]**2 + l_S[1,i]**2) + 1/(l_D[0]**2 + l_D[1]**2)) * (R[i]**2)
#     return SED_square, l_D


def find_near_and_far_points(scan_dir, src_obj, K, p_S, near_quantile=0.1, far_quantile=0.9):
    frame_name = src_obj['frame_name']
    dimension = src_obj['dimension']

    depth_img_path = os.path.join(scan_dir, 'depth', '{0}.png'.format(frame_name))
    depth_img = np.array(Image.open(depth_img_path))

    meta_file_path = os.path.join(scan_dir, '{0}.txt'.format(scan_dir.split('/')[-1]))
    depth_intrinsic = utils.read_depth_intrinsic(meta_file_path)
    color2depth_extrinsic = utils.read_color2depth_extrinsic(meta_file_path)

    pts_camera = utils.cropped_depth_to_point_cloud(depth_img, depth_intrinsic, dimension)
    # in case there is no points after filtering out
    if pts_camera.shape[0] == 0: return None

    pts_camera_ext = np.ones((pts_camera.shape[0], 4))
    pts_camera_ext[:, 0:3] = pts_camera[:, 0:3]

    if color2depth_extrinsic is not None:
        pts_camera_ext = np.dot(pts_camera_ext, color2depth_extrinsic.transpose())

    depths = pts_camera_ext[:,2]
    # select the q-th quantile of the depth as the near/far, in order to eliminate the outliers
    depths = np.sort(depths)
    z_near = np.quantile(depths, near_quantile)
    z_far = np.quantile(depths, far_quantile)

    pts = np.linalg.inv(K) @ p_S
    pts_near = pts * z_near
    pts_far = pts * z_far

    return (pts_near, pts_far)


def compute_epipolar_line_segment(scan_dir, src_obj, poses, K, src_frame_idx, dst_frame_idx, p_S):
    # pts_near and pts_far are in the src_camera coordinate
    pts = find_near_and_far_points(scan_dir, src_obj, K, p_S)
    if pts is None: return None
    pts_near, pts_far = pts

    # convert them into the dst_camera coordinate and project to the dst_frame
    relative_pose = np.dot(np.linalg.inv(poses[dst_frame_idx]), poses[src_frame_idx])  # (4,4)
    R = relative_pose[:3, :3]  # (3*3)
    t = relative_pose[:3, 3]  # (3,)
    A = K @ (R @ pts_near + t.reshape(-1,1))
    B = K @ (R @ pts_far + t.reshape(-1,1))
    A /= A[2, 0]
    B /= B[2, 0]

    return A, B


# def segment_intersect_rectangle(rect_minX, rect_minY, rect_maxX, rect_maxY, A, B):
#     #find min and max X for the line segment
#     minX = A[0]
#     maxX = B[0]
#     if A[0] > B[0]:
#         minX = B[0]
#         maxX = A[0]
#
#     # find the intersection of the line segments and rectangle's x-projections
#     if maxX > rect_maxX: maxX = rect_maxX
#     if minX < rect_minX: minX = rect_minX
#
#     # if not intersect return False
#     if minX > maxX: return False
#
#     # find corresponding min and max Y for min and max X we found before
#     minY = A[1]
#     maxY = B[1]
#
#     dx = B[0] - A[0]
#
#     if math.fabs(dx) > 0.0000001:
#         a = (B[1] - A[1])/dx
#         b = A[1] - a * A[0]
#         minY = a * minX + b
#         maxY = a * maxX + b
#
#     if minY > maxY:
#         tmp = maxY
#         maxY = minY
#         minY = tmp
#
#     # find the intersection of the segment's and rectangle's y-projections
#     if maxY > rect_maxY: maxY = rect_maxY
#     if minY < rect_minY: minY = rect_minY
#
#     # if not intersect return False
#     if minY > maxY: return False
#
#     return True


# def point_to_line_distance(A, B, P):
#     '''
#     :param A: endpoint of line segment
#     :param B: endpoint of line segment
#     :param P: point
#     :return: D: distance between point and line segment
#     '''
#     num = math.fabs((B[1]-A[1])*P[0] - (B[0]-A[0])*P[1] + B[0]*A[1] - B[1]*A[0])
#     denom = math.sqrt((B[1]-A[1])**2 + (B[0]-A[0])**2)
#     D = num / denom
#     return D


def point_to_line_distance(A, B, P):
    '''
    Calculate the distance between a point and a line segment.

    To calculate the closest distance to a line segment, we first need to check
    if the point projects onto the line segment.  If it does, then we calculate
    the orthogonal distance from the point to the line.
    If the point does not project to the line segment, we calculate the
    distance to both endpoints and take the shortest distance.

    :param A: endpoint of line segment (x1, y1)
    :param B: endpoint of line segment (x2, y2)
    :param P: point (x, y)
    :return: D: distance between point and line segment
    '''
    # dot = (x-x1)*(x2-x1) + (y-y1)*(y2-y1)
    dot = (P[0]-A[0]) * (B[0]-A[0]) + (P[1]-A[1]) * (B[1]-A[1])
    len_sq = (B[0]-A[0])**2 + (B[1]-A[1])**2
    param = -1
    # in case of 0 length line
    if len_sq != 0: param = dot/len_sq

    if param<0:
        xx = A[0]
        yy = A[1]
    elif param>1:
        xx = B[0]
        yy = B[1]
    else:
        xx = A[0] + param * (B[0]-A[0])
        yy = A[1] + param * (B[1]-A[1])

    D = math.sqrt((P[0]-xx)**2 + (P[1]-yy)**2)

    return D


def pairwise_association(scan_dir, frame_names, objects_index, objects, poses, K, MIN_scale=0.4, MIN_DIST=100,
                         to_visualize_epipolar=False, to_visualize_traj=False):
    '''
    Get the continuous tracking for objects by checking the occurrence of one object in the pairwise frame.
    :param scan_dir: the path to the scan data
    :param frame_names: list of frame names, which to access the data
    :param objects_index: list of (frame_idx, obj_idx)
    :param objects: list of object list in each frame, [[obj1, obj2,...],[obj1, obj2, ...],[obj1...]]
    :param poses: list of pose (camera extrinsic parameters) for each frame
    :param K: camera intrinsic parameter for this scan
    :param MIN_scale: the minimum size ratio between candidate bbox and the source bbox
    :param MIN_DIST: the minimum distance in checking epipolar constraint
    :param to_visualize_epipolar: to visualize the point and its epipolar line segment in a pair of frames
    :param to_visualize_traj: to visualize the tracking results by pairwise association.
    :return: trajectories, list of (classname, trajectory), where trajectory is a list of (frame_idx, obj_idx)
    '''

    def _compute_size_ratio(src_frame_idx, src_bbox_idx, dst_frame_idx, dst_bbox_idx):
        boxA = objects[src_frame_idx][src_bbox_idx]['dimension']
        boxB = objects[dst_frame_idx][dst_bbox_idx]['dimension']

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the scale of the bigger box over smaller box
        scale = boxBArea / boxAArea if boxAArea > boxBArea else boxAArea / boxBArea
        return scale

    def _check_epipolar_constraint(src_obj, dst_candidate_objs, src_frame_idx, dst_frame_idx):
        src_center = src_obj['center']
        p_S = np.ones((3, 1))
        p_S[:2, 0] = src_center

        # compute the two end points A and B for epipolar line segment
        line_segment = compute_epipolar_line_segment(scan_dir, src_obj, poses, K, src_frame_idx, dst_frame_idx, p_S)
        if line_segment is None: return None
        A, B = line_segment

        # if the A or B position out of the image, return None
        if A[0] < 0 or A[0] > cfg.SCANNET.IMAGE_WIDTH or A[1] < 0 or A[1] > cfg.SCANNET.IMAGE_HEIGHT: return None
        if B[0] < 0 or B[0] > cfg.SCANNET.IMAGE_WIDTH or B[1] < 0 or B[1] > cfg.SCANNET.IMAGE_HEIGHT: return None

        dist = np.zeros((len(dst_candidate_objs)))
        for i, obj in enumerate(dst_candidate_objs):
            dist[i] = point_to_line_distance(A, B, obj['center'])

        if np.amin(dist) < MIN_DIST:
            dst_bbox_id = dst_candidate_objs[np.argsort(dist)[0]]['bbox_id']
            return ((A, B), dst_bbox_id)
        else:
            return None

    trajectories = []
    MAX_FRAME_IDX = len(frame_names) - 1

    while objects_index != []:
        cur_frame_idx = objects_index[0][0]

        # if iterate to the last frame in this scan, save all the objects in objects_index individually, end the loop
        if cur_frame_idx == MAX_FRAME_IDX:
            for (frame_idx, bbox_idx) in objects_index:
                obj = objects[frame_idx][bbox_idx]
                classname =  obj['classname']
                trajectory = [(frame_idx, bbox_idx)]
                trajectories.append((classname, trajectory))
            break

        cur_bbox_idx = objects_index[0][1]
        # print('\tTrajectory tracking for frame_{0}-Object{1}...'.format(cur_frame_idx, cur_bbox_idx))
        cur_obj = objects[cur_frame_idx][cur_bbox_idx]
        cur_classname = cur_obj['classname']
        trajectory = [(cur_frame_idx, cur_bbox_idx)]
        objs_to_delete = []
        src_frame_idx = cur_frame_idx
        src_bbox_idx = cur_bbox_idx

        for i in range(len(frame_names)-cur_frame_idx):

            dst_frame_idx = src_frame_idx + 1
            if dst_frame_idx > MAX_FRAME_IDX: break
            # print('\t\t---check with {0}-th frame---'.format(dst_frame_idx))

            dst_candidate_objs = []
            for dst_obj in objects[dst_frame_idx]:
                # check if there is any object with the same classname as src_bbox..
                if dst_obj['classname'] != cur_classname: continue
                # check if the dst_obj is still in the object_index list, not be linked by other trajectory..
                if (dst_frame_idx, dst_obj['bbox_id']) not in objects_index: continue
                dst_candidate_objs.append(dst_obj)

            if len(dst_candidate_objs) != 0:
                # forward check
                forward_check_result = _check_epipolar_constraint(objects[src_frame_idx][src_bbox_idx],
                                                                 dst_candidate_objs,
                                                                 src_frame_idx, dst_frame_idx)

                if forward_check_result is not None:
                    epipolar_line_segment_forward, dst_bbox_idx = forward_check_result
                    # check the size ratio between candidate bbox and the source bbox
                    scale = _compute_size_ratio(src_frame_idx, src_bbox_idx, dst_frame_idx, dst_bbox_idx)
                    if scale > MIN_scale:
                        # backward check
                        backward_check_result = _check_epipolar_constraint(objects[dst_frame_idx][dst_bbox_idx],
                                                                          objects[src_frame_idx],
                                                                          dst_frame_idx, src_frame_idx)

                        if backward_check_result is not None:
                            epipolar_line_segment_backward, ref_bbox_idx = backward_check_result

                            if src_bbox_idx == ref_bbox_idx:
                                # Visualize
                                if to_visualize_epipolar:
                                    visualize_epipolar_geometry(scan_dir, objects, src_frame_idx, src_bbox_idx,
                                                                epipolar_line_segment_forward, dst_frame_idx,
                                                                dst_bbox_idx, epipolar_line_segment_backward)

                                trajectory.append((dst_frame_idx, dst_bbox_idx))
                                objs_to_delete.append((src_frame_idx, src_bbox_idx))
                                src_frame_idx = dst_frame_idx
                                src_bbox_idx = dst_bbox_idx
                                continue

            objs_to_delete.append((src_frame_idx, src_bbox_idx))
            break

        #delete the tracked objects from objects_index
        for obj_to_delete in objs_to_delete:
            objects_index.remove(obj_to_delete)

        # we also keep the single frame as a trajecory
        trajectories.append((cur_classname, trajectory))

    if to_visualize_traj:
        # visualize via video
        trajectories_to_show = []
        for i, (classname, trajectory) in enumerate(trajectories):
            trajectories_to_show.append(trajectory)
        visualize_trajectory_in_videos(scan_dir, frame_names, objects, trajectories_to_show)

        # visualize via images
        # for i, (classname, trajectory) in enumerate(trajectories):
            # visualize_trajectory(scan_dir, objects, trajectory)

    return trajectories



def exhaustive_association(scan_dir, trajectories, frame_names, objects, poses, K, MIN_DIST=100, MAX_PAIR_ANGLE=110,
                           MIN_SIZE_RATIO=0.5, MIN_TRACK_LENGTH=3, MIN_COVERAGE_ANGLE=30, to_visualize_epipolar=False,
                           to_visualize_rotation=False, to_visualize_traj=False):
    '''
    Exhaustively and hierarchically check every two tracks (generated by pairwise association or merge in
    this association phase) for each class
    :param scan_dir: the path to the scan data
    :param trajectories: tracks generated by pairwise association
    :param frame_names: list of frame names, which to access the data
    :param objects: list of object list in each frame, [[obj1, obj2,...],[obj1, obj2, ...],[obj1...]]
    :param poses: list of pose (camera extrinsic parameters) for each frame
    :param K: camera intrinsic parameter for this scan
    :param MIN_DIST: the minimum distance threshold in checking epipolar constraint
    :param MAX_PAIR_ANGLE: the maximum angle threshold to determine if pair two frames
    :param MIN_SIZE_RATIO: the minimum size ratio to select the valid objects in one tracklet
    :param MIN_TRACK_LENGTH: the minimum length of tracklet to keep
    :param MIN_COVERAGE_ANGLE: the minimum coverage angle for filtering tracklet
    :param to_visualize_epipolar: to visualize the point and its epipolar line segment in a pair of frames
    :param to_visualize_rotation: to visualize the relative rotation angle between two frames.
    :param to_visualize_traj: to visualize the tracking results by exhaustive association.
    :return: full_trajectories_new, list of trajectories, where each trajectory is a list of (frame_idx, obj_idx)
    '''

    def _calc_forward_backward_DIST(src_obj_index, dst_obj_index):
        src_frame_idx = src_obj_index[0]
        src_bbox_idx = src_obj_index[1]
        dst_frame_idx = dst_obj_index[0]
        dst_bbox_idx = dst_obj_index[1]

        src_obj = objects[src_frame_idx][src_bbox_idx]
        dst_obj = objects[dst_frame_idx][dst_bbox_idx]

        p_S = np.ones((3, 1))
        p_S[:2, 0] = src_obj['center']
        p_D = np.ones((3, 1))
        p_D[:2, 0] = dst_obj['center']

        # epipolar_line_segment_forward
        forward_epipolar_line_segment = compute_epipolar_line_segment(scan_dir, src_obj, poses, K, src_frame_idx,
                                                                      dst_frame_idx, p_S)
        if forward_epipolar_line_segment is not None:
            A, B = forward_epipolar_line_segment
        else:
            return None
        if A[0] < 0 or A[0] > cfg.SCANNET.IMAGE_WIDTH or A[1] < 0 or A[1] > cfg.SCANNET.IMAGE_HEIGHT: return 1000000
        if B[0] < 0 or B[0] > cfg.SCANNET.IMAGE_WIDTH or B[1] < 0 or B[1] > cfg.SCANNET.IMAGE_HEIGHT: return 1000000

        forward_dist = point_to_line_distance(A, B, p_D)

        # epipolar_line_segment_backward
        backward_epipolar_line_segment = compute_epipolar_line_segment(scan_dir, dst_obj, poses, K, dst_frame_idx,
                                                                       src_frame_idx, p_D)
        if backward_epipolar_line_segment is not None:
            C, D = backward_epipolar_line_segment
        else:
            return None
        backward_dist = point_to_line_distance(C, D, p_S)

        # dist = forward_dist if forward_dist >= backward_dist else backward_dist
        # print('forward_dist:{0}; backward_dist:{1}'.format(forward_dist, backward_dist))
        dist = (forward_dist + backward_dist)/2

        if to_visualize_epipolar and 100< dist < MIN_DIST:
            visualize_epipolar_geometry(scan_dir, objects, src_frame_idx, src_bbox_idx, (A,B),
                                        dst_frame_idx, dst_bbox_idx,(C,D))

        return dist

    def _check_relative_rotation_constraint(traj_a, traj_b):
        '''
        :param traj_a: list of (frame_idx, bbox_idx) pairs in one trajectory
        :param traj_b: list of (frame_idx, bbox_idx) pairs in another trajectory
        :return: True/False
        '''
        # get the relative rotation matrix from the last frame in traj_a to the first frame in traj_b
        frameid_a = traj_a[-1][0]
        frameid_b = traj_b[0][0]

        angle = _calculate_relative_angle(poses[frameid_a][:3, :3], poses[frameid_b][:3, :3])
        if angle > MAX_PAIR_ANGLE:
            if to_visualize_rotation:
                # visualize the relative rotation angle between two frames in trajectory pair
                visualize_rotation_angle(scan_dir, objects, traj_a, traj_b, angle)
            return False
        else:
            return True

    def traj_pairwise_association(trajectories_to_link):

        trajectories_linked = []

        # extract trajectory pairs,
        # note that if the two trajectories are from the same frame, it means they are the two instances
        pairs = []
        for a, traj_a in enumerate(trajectories_to_link):
            frameids_a = set([t[0] for t in traj_a])
            for b, traj_b in enumerate(trajectories_to_link):
                if a == b: continue
                frameids_b = set([t[0] for t in traj_b])
                if frameids_a.isdisjoint(frameids_b) and _check_relative_rotation_constraint(traj_a, traj_b):
                    pairs.append((a,b))

        if pairs == []:
            return trajectories_to_link

        # compute initial algebraic distance for each trajectory pair
        M = {}
        for (a, b) in pairs:
            # check the algebraic distance between the last object in src_traj and the first object in dst_traj
            src_obj_index = trajectories_to_link[a][-1] #<frame_idx, bbox_idx>
            dst_obj_index = trajectories_to_link[b][0]
            dist = _calc_forward_backward_DIST(src_obj_index, dst_obj_index)
            if dist is not None:
                M[(a, b)] = dist

                # hierarchal search
        while M != {}:
            #select the pair with shortest distance
            sel_pair = sorted(M.items(), key=lambda i:i[1])[0]

            # if the distance is larger than the threshold, save the trajectories in M
            # elif the distance is less than the threshold, link the two trajectories
            if sel_pair[1] > MIN_DIST: break

            a, b = sel_pair[0]

            #link the two trajectory and put into the traj_linked
            new_traj = []
            new_traj.extend(trajectories_to_link[a])
            new_traj.extend(trajectories_to_link[b])
            trajectories_to_link.append(new_traj)

            ##make sure the trajectory across frames
            assert len(set(new_traj)) == len(new_traj)

            #delete (a,j), (i,b), (b,a) from M
            keys_to_delete = []
            for key in M.keys():
                if a==key[0] or b==key[1] or (a==key[1] and b==key[0]):
                    keys_to_delete.append(key)

            for k in keys_to_delete: del M[k]

            # if all the trajectories are pair up, add the new trajectory to list and break the loop
            if M == {}:
                trajectories_linked.append(new_traj)
                break

            # update (b,j) to (ab, j) and (i,a) to (i, ab)... if the frameids in i-th trajectory or j-th trajectory
            # conflicts with the frameids in ab-th trajectory, delete (i,a) or (b,j) from M...
            new_id = len(trajectories_to_link)-1
            keys_to_check = copy.deepcopy(list(M.keys()))
            frameids_new = set([t[0] for t in new_traj])
            for key in keys_to_check:
                if a in list(key):
                    frameids_to_check = set([t[0] for t in trajectories_to_link[key[0]]])
                    if frameids_new.isdisjoint(frameids_to_check):
                        M[(key[0], new_id)] = M.pop(key)
                    else:
                        M.pop(key)
                elif b in list(key):
                    frameids_to_check = set([t[0] for t in trajectories_to_link[key[1]]])
                    if frameids_new.isdisjoint(frameids_to_check):
                        M[(new_id, key[1])] = M.pop(key)
                    else:
                        M.pop(key)

        for (i,j) in M.keys():
            if trajectories_to_link[i] not in trajectories_linked:
                trajectories_linked.append(trajectories_to_link[i])
            if trajectories_to_link[j] not in trajectories_linked:
                trajectories_linked.append(trajectories_to_link[j])

        return trajectories_linked

    def select_valid_objects_in_one_track(tracklet):
        # filter out the bbox that lower than certain ratio of the average size of its trajectory
        objs_size = []
        for frame_idx, bbox_idx in tracklet:
            obj = objects[frame_idx][bbox_idx]
            size = (obj['dimension'][2] - obj['dimension'][0]) * (obj['dimension'][3] - obj['dimension'][1])
            objs_size.append(size)
        objs_size = np.array(objs_size)
        avg_objs_size = np.mean(objs_size)
        mask = objs_size > avg_objs_size * MIN_SIZE_RATIO
        # if np.sum(mask) != objs_size.shape[0]:
        #     print('to debug')
        return mask

    def valid_rotation_coverage(tracklet):
        '''compute the maximum rotation angle across all the pairs in one tracklet,
           which is considered as the rotation coverage of the tracklet.'''
        iterable = list(range(len(tracklet)))
        index_pairs_pool = list(itertools.combinations(iterable, 2))
        angle_pairs_pool = []
        for a, b in index_pairs_pool:
            frameid_a = tracklet[a][0]
            frameid_b = tracklet[b][0]
            angle = _calculate_relative_angle(poses[frameid_a][:3, :3], poses[frameid_b][:3, :3])
            angle_pairs_pool.append(angle)
        if max(angle_pairs_pool) > MIN_COVERAGE_ANGLE:
            return True
        else:
            return False

    class2traj = {}
    for i, (classname, trajectory) in enumerate(trajectories):
        if classname in class2traj.keys():
            class2traj[classname].append(i)
        else:
            class2traj[classname] = [i]

    full_trajectories = []
    for classname, traj_ids in class2traj.items():
        if len(traj_ids) == 1:
            # if there is only one trajectory in this class, save it
            traj_id = traj_ids[0]
            full_trajectories.append(trajectories[traj_id][1])
        else:
            # if there are multiple trajectories, link them via pairwise association
            trajectories_to_link = [trajectories[traj_id][1] for traj_id in traj_ids]
            trajectories_linked = traj_pairwise_association(trajectories_to_link)
            full_trajectories.extend(trajectories_linked)

    # visualize_trajectory_in_videos(scan_dir, frame_names, objects, full_trajectories)

    # filter out the trajectories less than a certain number (ie 3) if frames..
    full_trajectories_new = []
    for full_trajectory in full_trajectories:
        if len(full_trajectory) >= MIN_TRACK_LENGTH:
            mask = select_valid_objects_in_one_track(full_trajectory)
            selected_trajectory = np.array(full_trajectory)[mask]
            if valid_rotation_coverage(selected_trajectory) and len(selected_trajectory) >= MIN_TRACK_LENGTH:
                full_trajectories_new.append(selected_trajectory)

    if to_visualize_traj:
        # visualize via video
        visualize_trajectory_in_videos(scan_dir, frame_names, objects, full_trajectories_new)
        ##visualize via images
        #for i, full_trajectory in enumerate(full_trajectories_new):
        #    visualize_trajectory(scan_dir, objects, full_trajectory)

    return full_trajectories_new


def tracking(scan_dir, scan_name, valid_frame_names, min_ratio=9, pairwise_min_scale=0.4, pairwise_min_dist=50,
             visu_pairwise_epipolar=False, visu_pairwise_traj=True, exhaustive_min_dist=100,
             exhaustive_max_pair_angle=110, exhausive_min_size_ratio=0.5, min_track_length=3, min_coverage_angle=30,
             visu_exhaustive_epipolar=False, visu_exhaustive_rotation=False, visu_exhaustive_traj=True):
    meta_file = os.path.join(scan_dir, '{0}.txt'.format(scan_name))
    K = utils.read_camera_intrinsic(meta_file)[:,:3] #(3,3)

    # Use the uncalibrated intrinsic matrix
    # camera_intrinsic_path = os.path.join(scan_dir, 'intrinsic', 'intrinsic_color.txt')
    # K = np.loadtxt(camera_intrinsic_path)[:3, :3]

    frame_names = []
    objects = []
    objects_index = []
    poses = []

    frame_idx = 0
    for frame_name in valid_frame_names:
        #TODO: modify file name after new data comes
        label_file = os.path.join(scan_dir, 'bbox2d_18class', '{0}_bbox.pkl'.format(frame_name))
        # label_file = os.path.join(scan_dir, 'bbox2d_{0}class'.format(len(cfg.SCANNET.CLASSES)),
        #                                     '{0}_bbox.pkl'.format(frame_name))
        bboxes2d = utils.read_2Dbbox(label_file)  # list of dictionaries

        ## filter out some invalid bbox (e.g., the width or height or area of bbox is lower than a threshold..)
	    # TODO: consider prior size of each class to filter out relatively small bbox..
        new_bboxes2d = []
        bbox_idx = 0
        for i, bbox in enumerate(bboxes2d):
            dimension = bbox['box2d']
            width = dimension[2] - dimension[0]
            height = dimension[3] - dimension[1]
            if width > (cfg.SCANNET.IMAGE_WIDTH / min_ratio) and height > (cfg.SCANNET.IMAGE_HEIGHT / min_ratio):
                new_bboxes2d.append({'bbox_id': bbox_idx,
                                     'instance_id': bbox['instance_id'],
                                     'classname': bbox['classname'],
                                     'center': [(dimension[0]+dimension[2])/2.0, (dimension[1]+dimension[3])/2.0],
                                     'dimension': dimension,
                                     'frame_name': frame_name
                                     })
                objects_index.append((frame_idx, bbox_idx))
                bbox_idx += 1
        if len(new_bboxes2d) > 0:
            objects.append(new_bboxes2d)
            frame_names.append(frame_name)
            pose_file = os.path.join(scan_dir, 'pose', '{0}.txt'.format(frame_name))
            pose_inv = np.loadtxt(pose_file)  # 4*4, the matrix maps the camera coord to world coord
            poses.append(pose_inv)
            frame_idx += 1

    print('Filter out bboxes whose width or height is less than 1/{0} of that of image...\n '
          '\t[original frames] - {1}; [remaining frames] - {2}'.format(min_ratio, len(valid_frame_names),
                                                                        len(frame_names)))

    start = timer()
    trajectories = pairwise_association(scan_dir, frame_names, objects_index, objects, poses, K, pairwise_min_scale,
                                        pairwise_min_dist, visu_pairwise_epipolar, visu_pairwise_traj)
    print('Got {0} trajectories via pairwise association, timer--{1}'.format(len(trajectories), timer()-start))

    start = timer()
    full_trajectories = exhaustive_association(scan_dir, trajectories, frame_names, objects, poses, K,
                                               exhaustive_min_dist, exhaustive_max_pair_angle, exhausive_min_size_ratio,
                                               min_track_length, min_coverage_angle, visu_exhaustive_epipolar,
                                               visu_exhaustive_rotation, visu_exhaustive_traj)
    print('Got {0} trajectories after exhaustive association, timer--{1}'.format(len(full_trajectories), timer()-start))

    return objects, full_trajectories



def main(opt):
    data_dir = os.path.join(opt.root_dir, 'scans')

    split_file = os.path.join(opt.root_dir, 'traintestsplit', 'scannetv2_train.txt')
    scan_name_list = [x.strip() for x in open(split_file).readlines()]
    print('[ScanNet, Train] - {0} samples\n'.format(len(scan_name_list)))

    if opt.scan_id:
        if opt.scan_id in scan_name_list:
            scan_name_list = [opt.scan_id]
        else:
            print('ERROR: Invalid scan id: ' + opt.scan_id)
    else:
        # shuffle the list for debugging
        #TODO: remove this after debugging
        random.shuffle(scan_name_list)

    for scan_idx, scan_name in enumerate(scan_name_list):
        print('-----------Process ({0}, {1})-----------'.format(scan_idx, scan_name))
        scan_dir = os.path.join(data_dir, scan_name)
        valid_frame_names_file = os.path.join(scan_dir, '{0}_validframes_18class_{1}frameskip.txt'
                                            .format(scan_name, cfg.SCANNET.FRAME_SKIP))
        valid_frame_names = [int(x.strip()) for x in open(valid_frame_names_file).readlines()]

        objects, trajectories = tracking(scan_dir, scan_name, valid_frame_names)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/mnt/Data/Datasets/ScanNet_v2/', help='path to data')
    parser.add_argument('--scan_id', default=None, help='specific scan id to download') #scene0463_01, 'scene0130_00'

    opt = parser.parse_args()

    main(opt)
