#!/usr/bin/python3
import sys
sys.path.append('../')
import os
import math
import random
import numpy as np
import cv2
from PIL import Image
# import matplotlib
# matplotlib.use('TkAGG')
from matplotlib import pyplot as plt

import data.scannet.scannet_utils as utils
from config import cfg


def fundamental_matrix(K, poses, src_frame_idx, dst_frame_idx):
    relative_pose = np.dot(np.linalg.inv(poses[dst_frame_idx]), poses[src_frame_idx])  # (4,4)
    R = relative_pose[:3, :3]  # (3*3)
    t = relative_pose[:3, 3]  # (3,)
    t_x = np.array([[0, -t[2], t[1]],
                    [t[2], 0, -t[0]],
                    [-t[1], t[0], 0]])
    E = np.dot(t_x, R)
    F = np.linalg.inv(K.transpose()).dot(E).dot(np.linalg.inv(K))
    return F


def algebraic_distance(poses, K, src_frame_idx, dst_frame_idx, p_S, p_D):
    '''
    --> refer to Section 4.2 in 'Fundamental Matrix Estimation:  A Study of Error Criteria'
        algebraic distance R = R(x_s, x_d) = (x_d)^T * F * (x_s)
    :param p_S: point in image I_S, numpy array, (3,1)
    :param p_D: point(s) in image I_D, numpy array, (3,n)
             I_S and I_D are two perspective images of the same scene
    :return: R: algebraic epipolar distance:  scalar
             l_D: epipolar line in I_D that corresponds to x_s
             l_S: epipolar line in I_S that corresponds to x_d
    '''
    F = fundamental_matrix(K, poses, src_frame_idx, dst_frame_idx)  # (3,3)
    l_D = np.dot(F, p_S)  # epipolar line (3,1)
    l_S = np.dot(F.transpose(), p_D) #epipolar lines (3, n)
    R = np.dot(p_D.transpose(), l_D).reshape(-1)  # (n)
    return R, l_D, l_S


def symmetric_epipolar_distance(poses, K, src_frame_idx, dst_frame_idx, p_S, p_D):
    '''
    --> refer to Section 4.3 in 'Fundamental Matrix Estimation:  A Study of Error Criteria'
        It measures the geometric distance of each point to it sepipolar line

    :param p_S: point in image I_S, numpy array, (3,1)
    :param p_D: point(s) in image I_D, numpy array, (3,n)
             I_S and I_D are two perspective images of the same scene
    :return: SED_square: symmtric epipolar distance: numpy array, each element is a non-negative scalar
             l: epipolar line
    '''
    R, l_D, l_S = algebraic_distance(poses, K, src_frame_idx, dst_frame_idx, p_S, p_D)
    n_pts = p_D.shape[1]
    SED_square = np.zeros((n_pts))
    for i in range(n_pts):
        SED_square[i] = (1/(l_S[0,i]**2 + l_S[1,i]**2) + 1/(l_D[0]**2 + l_D[1]**2)) * (R[i]**2)
    return SED_square, l_D


def check_epipolar_constraint(src_obj, dst_candidate_objs, poses, K, src_frame_idx, dst_frame_idx, MIN_SED=10000.0):

    src_center = src_obj['center']
    p_S = np.ones((3,1))
    p_S[:2, 0] = src_center

    p_D = np.ones((3, len(dst_candidate_objs))) #(3,n)
    for i, obj in enumerate(dst_candidate_objs):
        p_D[:2, i] = obj['center']

    SED, epipolar_line = symmetric_epipolar_distance(poses, K, src_frame_idx, dst_frame_idx, p_S, p_D)

    if np.amin(SED) < MIN_SED:
        sel_ids = np.where(SED<MIN_SED)[0]
        if len(sel_ids) > 1:
            sel_dist = SED[sel_ids]
            sel_ids = sel_ids[np.argsort(sel_dist)]
        dst_bbox_ids = [dst_candidate_objs[sel_id]['bbox_id'] for sel_id in sel_ids]
        return (dst_bbox_ids, epipolar_line)
    else:
        return None


# def compute_IoU_and_scale(objects, src_frame_idx, src_bbox_idx, dst_frame_idx, dst_bbox_idx):
#     boxA = objects[src_frame_idx][src_bbox_idx]['dimension']
#     boxB = objects[dst_frame_idx][dst_bbox_idx]['dimension']
#
#     # determine the (x, y)-coordinates of the intersection rectangle
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])
#
#     # compute the area of intersection rectangle
#     interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
#
#     # compute the area of both the prediction and ground-truth rectangles
#     boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
#     boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
#
#     # compute the intersection over union by taking the intersection area and dividing it by union area
#     iou = interArea / float(boxAArea + boxBArea - interArea)
#
#     # compute the scale of the bigger box over smaller box
#     scale = boxBArea / boxAArea if boxAArea > boxBArea else boxAArea / boxBArea
#
#     return iou, scale


def upsize_box(box, upsize_times):
    w = box[2] - box[0]
    h = box[3] - box[1]

    offset_w = (upsize_times-1) * w
    offset_h = (upsize_times-1) * h

    new_box = [0, 0, cfg.SCANNET.IMAGE_WIDTH, cfg.SCANNET.IMAGE_HEIGHT]
    new_box[0] = max(new_box[0], box[0]-(offset_w//2))
    new_box[1] = max(new_box[1], box[1]-(offset_h//2))
    new_box[2] = min(new_box[2], box[2]+(offset_w-offset_w//2))
    new_box[3] = min(new_box[3], box[3]+(offset_h-offset_h//2))

    return new_box


def compute_spatial_overlap(boxA, boxB, boxAArea, boxBArea, MIN_BOX_RATIO=8, UPSIZE_TIMES=1.5):
    '''
    We assume that the positions of one object in two consecutive frames should have an overlap
    In case the size of the object is too small, we also enlarge the object size to some extent.
    :return: overlap ratio
    '''
    # TODO: measure the size of object, enlarge the bbox n times if the size is lower than a threshold.
    MIN_BOX_AREA = (cfg.SCANNET.IMAGE_HEIGHT // MIN_BOX_RATIO) * (cfg.SCANNET.IMAGE_WIDTH // MIN_BOX_RATIO)

    if boxAArea < MIN_BOX_AREA: boxA = upsize_box(boxA, UPSIZE_TIMES)
    if boxBArea < MIN_BOX_AREA: boxB = upsize_box(boxB, UPSIZE_TIMES)

     # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection area and dividing it by union area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def compute_size_ratio(objects, src_frame_idx, src_bbox_idx, dst_frame_idx, dst_bbox_idx):
    boxA = objects[src_frame_idx][src_bbox_idx]['dimension']
    boxB = objects[dst_frame_idx][dst_bbox_idx]['dimension']

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the scale of the bigger box over smaller box
    scale = boxBArea / boxAArea if boxAArea > boxBArea else boxAArea / boxBArea
    return scale


def compute_scale_and_IoU(objects, src_frame_idx, src_bbox_idx, dst_frame_idx, dst_bbox_idx):
    boxA = objects[src_frame_idx][src_bbox_idx]['dimension']
    boxB = objects[dst_frame_idx][dst_bbox_idx]['dimension']

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    scale = boxBArea / boxAArea if boxAArea > boxBArea else boxAArea / boxBArea

    iou = compute_spatial_overlap(boxA, boxB, boxAArea, boxBArea)

    return scale, iou


def visualize_epipolar_geometry(scan_dir, objects, src_frame_idx, src_bbox_idx, epipolar_line_forward, dst_frame_idx,
                                dst_bbox_ids, sel_dst_bbox_idx, epipolar_line_backward, ref_bbox_ids):
    src_obj = objects[src_frame_idx][src_bbox_idx]
    src_img = cv2.imread(os.path.join(scan_dir, 'color', '{0}.jpg'.format(src_obj['frame_id'])))
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

    dst_obj = objects[dst_frame_idx][dst_bbox_ids[0]]
    dst_img = cv2.imread(os.path.join(scan_dir, 'color', '{0}.jpg'.format(dst_obj['frame_id'])))
    dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)

    r, c = src_img.shape[:2]
    img1 = src_img.copy()
    img2 = dst_img.copy()
    color = tuple(np.random.randint(0, 255, 3).tolist())
    x0, y0 = map(int, [0, -epipolar_line_forward[2] / epipolar_line_forward[1]])
    x1, y1 = map(int, [c, -(epipolar_line_forward[2] + epipolar_line_forward[0] * c) /
                       epipolar_line_forward[1]])
    cv2.line(img2, (x0, y0), (x1, y1), color, 10)
    for dst_bbox_idx in dst_bbox_ids:
        dst_obj = objects[dst_frame_idx][dst_bbox_idx]
        cv2.circle(img2, (int(dst_obj['center'][0]), int(dst_obj['center'][1])), 30, color, -1)
        cv2.putText(img2, '%s' % str(dst_bbox_idx),
                    (max(int(dst_obj['center'][0]), 15), max(int(dst_obj['center'][1]), 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 4)
    cv2.circle(img1, (int(src_obj['center'][0]), int(src_obj['center'][1])), 30, color, -1)
    plt.subplot(221), plt.imshow(img1)
    plt.subplot(222), plt.imshow(img2)

    img4 = dst_img.copy()
    img3 = src_img.copy()
    x2, y2 = map(int, [0, -epipolar_line_backward[2] / epipolar_line_backward[1]])
    x3, y3 = map(int, [c, -(epipolar_line_backward[2] + epipolar_line_backward[0] * c) /
                       epipolar_line_backward[1]])
    cv2.line(img3, (x2, y2), (x3, y3), color, 10)
    sel_dst_obj = objects[dst_frame_idx][sel_dst_bbox_idx]
    for ref_bbox_idx in ref_bbox_ids:
        ref_obj = objects[src_frame_idx][ref_bbox_idx]
        cv2.circle(img3, (int(ref_obj['center'][0]), int(ref_obj['center'][1])), 30, color, -1)
    cv2.circle(img4, (int(sel_dst_obj['center'][0]), int(sel_dst_obj['center'][1])), 30, color, -1)
    plt.subplot(223), plt.imshow(img3)
    plt.subplot(224), plt.imshow(img4)
    plt.show()
    # plt.waitforbuttonpress()


def visualize_trajectory(scan_dir, objects, trajectory):
    num_frames = len(trajectory)
    ncols = int(math.sqrt(num_frames))
    nrows = (num_frames // ncols) + 1

    for i, (frame_idx, bbox_idx) in enumerate(trajectory):
        obj = objects[frame_idx][bbox_idx]
        img_path = os.path.join(scan_dir, 'color', '{0}.jpg'.format(obj['frame_id']))

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cv2.rectangle(img, (int(obj['dimension'][0]), int(obj['dimension'][1])),
                           (int(obj['dimension'][2]), int(obj['dimension'][3])),
                            (0, 255, 0), 15)
        cv2.putText(img, '%d %s' % (obj['instance_id'], obj['classname']),
                    (max(int(obj['dimension'][0]), 15), max(int(obj['dimension'][1]), 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 5)
        plt.subplot(nrows, ncols, i+1), plt.imshow(img)
    plt.show()


def pairwise_association(scan_dir, frame_ids, objects_index, objects, poses, K, MIN_scale=0.5, MIN_IoU=0.05,
                         to_visualize_epipolar=False,  to_visualize_traj=False):
    '''
    Get the continuous tracking for objects by checking the occurrence of one object in the pairwise frame.
    :param scan_dir: the path to the scan data
    :param frame_ids: list of frame ID, which to access the data
    :param objects_index: list of (frame_idx, obj_idx)
    :param objects: list of object list in each frame, [[obj1, obj2,...],[obj1, obj2, ...],[obj1...]]
    :param poses: list of pose (camera extrinsic parameters) for each frame
    :param K: camera intrinsic parameter for this scan
    :param MIN_scale: the minimum size ratio between candidate bbox and the source bbox
    :param MIN_IoU: the minimum overlap between candidate bbox and the source bbox
    :param to_debug: to visualize the tracking results during debug.
    :return: trajectories, list of (classname, trajectory), where trajectory is a list of (frame_idx, obj_idx)
    '''
    trajectories = []
    MAX_FRAME_IDX = len(frame_ids) - 1

    while objects_index != []:
        cur_frame_idx = objects_index[0][0]

        # if iterate to the last frame in this scan, end the loop
        if cur_frame_idx == MAX_FRAME_IDX: break

        cur_bbox_idx = objects_index[0][1]
        print('\tTrajectory tracking for frame_{0}-Object{1}...'.format(cur_frame_idx, cur_bbox_idx))
        cur_obj = objects[cur_frame_idx][cur_bbox_idx]
        cur_classname = cur_obj['classname']
        trajectory = [(cur_frame_idx, cur_bbox_idx)]
        objs_to_delete = []
        src_frame_idx = cur_frame_idx
        src_bbox_idx = cur_bbox_idx

        for i in range(len(frame_ids)-cur_frame_idx):

            dst_frame_idx = src_frame_idx + 1
            if dst_frame_idx > MAX_FRAME_IDX: break
            print('\t\t---check with {0}-th frame---'.format(dst_frame_idx))

            dst_candidate_objs = []
            for dst_obj in objects[dst_frame_idx]:
                # check if there is any object with the same classname as src_bbox..
                if dst_obj['classname'] != cur_classname: continue
                # check if the dst_obj is still in the object_index list
                if (dst_frame_idx, dst_obj['bbox_id']) not in objects_index: continue
                dst_candidate_objs.append(dst_obj)

            if len(dst_candidate_objs) == 0:
                objs_to_delete.append((src_frame_idx, src_bbox_idx))
                break
            else:
                # forward check
                forward_check_result = check_epipolar_constraint(objects[src_frame_idx][src_bbox_idx],
                                                                 dst_candidate_objs,
                                                                 poses, K,
                                                                 src_frame_idx, dst_frame_idx,
                                                                 MIN_SED=8100.0)

                if forward_check_result == None:
                    objs_to_delete.append((src_frame_idx, src_bbox_idx))
                    break
                else:
                    dst_bbox_ids, epipolar_line_forward = forward_check_result
                    for dst_bbox_idx in dst_bbox_ids:
                        # check the size ratio between candidate bbox and the source bbox
                        scale = compute_size_ratio(objects, src_frame_idx, src_bbox_idx, dst_frame_idx, dst_bbox_idx)
                        if scale > MIN_scale:
                            # backward check
                            backward_check_result = check_epipolar_constraint(objects[dst_frame_idx][dst_bbox_idx],
                                                                              objects[src_frame_idx],
                                                                              poses, K,
                                                                              dst_frame_idx, src_frame_idx,
                                                                              MIN_SED=8100.0)

                            if backward_check_result != None:
                                ref_bbox_ids, epipolar_line_backward = backward_check_result

                                if src_bbox_idx in ref_bbox_ids:
                                    # Visualize
                                    if to_visualize_epipolar:
                                        visualize_epipolar_geometry(scan_dir, objects, src_frame_idx, src_bbox_idx,
                                                                    epipolar_line_forward, dst_frame_idx, dst_bbox_ids,
                                                                    dst_bbox_idx, epipolar_line_backward, ref_bbox_ids)

                                    trajectory.append((dst_frame_idx, dst_bbox_idx))
                                    objs_to_delete.append((src_frame_idx, src_bbox_idx))
                                    src_frame_idx = dst_frame_idx
                                    src_bbox_idx = dst_bbox_idx
                                    break
                    else:
                        objs_to_delete.append((src_frame_idx, src_bbox_idx))
                        break

        #delete the tracked objects from objects_index
        for obj_to_delete in objs_to_delete:
            objects_index.remove(obj_to_delete)

        if len(trajectory) > 1:
            #TODO: should we keep the single frame as a trajecory?
            trajectories.append((cur_classname, trajectory))

    if to_visualize_traj:
        for i, (classname, trajectory) in enumerate(trajectories):
            visualize_trajectory(scan_dir, objects, trajectory)

    return trajectories



def exhaustive_association(scan_dir, trajectories, objects, poses, K, to_visualize_epipolar=False,
                           to_visualize_traj=False, MIN_SED=10000.0):

    def _calc_forward_backward_SED(src_obj_index, dst_obj_index):
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

        forward_SED, forward_epipolar_line = symmetric_epipolar_distance(poses, K, src_frame_idx, dst_frame_idx,
                                                                          p_S, p_D)
        backward_SED, backward_epipolar_line = symmetric_epipolar_distance(poses, K, dst_frame_idx, src_frame_idx,
                                                                            p_D, p_S)
        SED = forward_SED if forward_SED >= backward_SED else backward_SED

        if to_visualize_epipolar and SED < MIN_SED:
            visualize_epipolar_geometry(scan_dir, objects, src_frame_idx, src_bbox_idx, forward_epipolar_line,
                                        dst_frame_idx, [dst_bbox_idx], dst_bbox_idx, backward_epipolar_line,
                                        [src_bbox_idx])

        return SED

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
                if frameids_a.isdisjoint(frameids_b):
                    pairs.append((a,b))

        if pairs == []:
            return trajectories_to_link

        # compute initial algebraic distance for each trajectory pair
        M = {}
        for (a, b) in pairs:
            # check the algebraic distance between the last object in src_traj and the first object in dst_traj
            src_obj_index = trajectories_to_link[a][-1] #<frame_idx, bbox_idx>
            dst_obj_index = trajectories_to_link[b][0]
            M[(a, b)] = _calc_forward_backward_SED(src_obj_index, dst_obj_index)

        # hierarchal search
        while M != {}:
            #select the pair with shortest distance
            sel_pair = sorted(M.items(), key=lambda i:i[1])[0]

            # if the distance is larger than the threshold, save the trajectories in M
            # elif the distance is less than the threshold, link the two trajectories
            if sel_pair[1] > MIN_SED: break

            a, b = sel_pair[0]

            #link the two trajectory and put into the traj_linked
            new_traj = []
            new_traj.extend(trajectories_to_link[a])
            new_traj.extend(trajectories_to_link[b])
            trajectories_to_link.append(new_traj)

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

            # update (b,j) to (ab, j) and (i,a) to (i, ab)
            new_id = len(trajectories_to_link)-1
            for key in M.keys():
                if a in key:
                    M[(key[0], new_id)] = M.pop(key)
                elif b in key:
                    M[(new_id, key[1])] = M.pop(key)

        for (i,j) in M.keys():
            if trajectories_to_link[i] in trajectories_linked or trajectories_to_link[j] in trajectories_linked: continue
            trajectories_linked.append(trajectories_to_link[i])
            trajectories_linked.append(trajectories_to_link[j])

        return trajectories_linked

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

    if to_visualize_traj:
        for i, full_trajectory in enumerate(full_trajectories):
            visualize_trajectory(scan_dir, objects, full_trajectory)

    return full_trajectories


def process_one_scan(scan_dir, scan_name, valid_frame_ids, min_ratio=10):
    meta_file = os.path.join(scan_dir, '{0}.txt'.format(scan_name))
    K = utils.read_camera_intrinsic(meta_file)[:,:3] #(3,3)

    # camera_intrinsic_path = os.path.join(scan_dir, 'intrinsic', 'intrinsic_color.txt')
    # K = np.loadtxt(camera_intrinsic_path)[:3, :3]

    frame_ids = []
    objects = []
    objects_index = []
    poses = []

    frame_idx = 0
    for frame_id in valid_frame_ids:
        label_file = os.path.join(scan_dir, 'bbox2d_18class', '{0}_bbox.pkl'.format(frame_id))
        bboxes2d = utils.read_2Dbbox(label_file)  # list of dictionaries

        ## filter out some invalid bbox (e.g., the width or height or area of bbox is lower than a threshold..)i
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
                                     'frame_id': frame_id
                                     })
                objects_index.append((frame_idx, bbox_idx))
                bbox_idx += 1
        if len(new_bboxes2d) > 0:
            objects.append(new_bboxes2d)
            frame_ids.append(frame_id)
            pose_file = os.path.join(scan_dir, 'pose', '{0}.txt'.format(frame_id))
            pose_inv = np.loadtxt(pose_file)  # 4*4, the matrix maps the camera coord to world coord
            poses.append(pose_inv)
            frame_idx += 1

    print('[original frames] - {0}; [remaining frames] - {1}'.format(len(valid_frame_ids), len(frame_ids)))

    trajectories = pairwise_association(scan_dir, frame_ids, objects_index, objects, poses, K,
                                        to_visualize_epipolar=False, to_visualize_traj=False)
    print('Got {0} trajectories via pairwise association'.format(len(trajectories)))

    full_trajectories = exhaustive_association(scan_dir, trajectories, objects, poses, K,
                                               to_visualize_epipolar=False, to_visualize_traj=True)
    print('Got {0} trajectories after exhaustive association'.format(len(full_trajectories)))



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
        random.shuffle(scan_name_list)


    for scan_idx, scan_name in enumerate(scan_name_list):
        print('-----------Process ({0}, {1})-----------'.format(scan_idx, scan_name))
        scan_dir = os.path.join(data_dir, scan_name)
        valid_frame_ids_file = os.path.join(scan_dir, '{0}_validframes_18class_{1}frameskip.txt'
                                            .format(scan_name, opt.frame_skip))
        valid_frame_ids = [int(x.strip()) for x in open(valid_frame_ids_file).readlines()]

        process_one_scan(scan_dir, scan_name, valid_frame_ids)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/mnt/Data/Datasets/ScanNet_v2/', help='path to data')
    parser.add_argument('--scan_id', default='scene0463_01', help='specific scan id to download') #scene0463_01
    parser.add_argument('--frame_skip', type=int, default=15,
                        help='the number of frames to skip in extracting instance annotation images')

    opt = parser.parse_args()

    main(opt)
