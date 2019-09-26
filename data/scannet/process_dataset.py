import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.transform import Rotation as Rot

from extract_bboxes2D import unzip_instace_file, get_objectID2label_and_color, extract_bbox2d_for_one_scan
import scannet_utils
from config import cfg


def visualize_rotation_angle(scan_dir, framename_a, framename_b, angle):
    '''visualize the relative rotation angle between two frames in trajectory pair'''
    src_img = cv2.imread(os.path.join(scan_dir, 'color', '{0}.jpg'.format(framename_a)))
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

    dst_img = cv2.imread(os.path.join(scan_dir, 'color', '{0}.jpg'.format(framename_b)))
    dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)

    plt.subplot(121), plt.imshow(src_img)
    plt.gca().set_title(framename_a)
    plt.subplot(122), plt.imshow(dst_img)
    plt.gca().set_title('{0}-{1}'.format(framename_b, angle))
    plt.show()


def compute_relative_rotation(src_pose, dst_pose):
    R_relative = (np.linalg.inv(dst_pose) @ src_pose)[:3, :3]
    r = Rot.from_dcm(R_relative)
    rot_vec = r.as_rotvec()
    angle = np.degrees(np.linalg.norm(rot_vec))
    return angle


def find_valid_pose(scan_path, frame_name):
    pose_file_path = os.path.join(scan_path, 'pose', '{0}.txt'.format(frame_name))
    if os.path.exists(pose_file_path):
        camera2world_extrinsic = np.loadtxt(pose_file_path)
        if not np.any(np.isinf(camera2world_extrinsic)):
            return True
    return False


def find_valid_objects(scan_path, frame_name, objectID2label, LABEL_MAP, TARGET_CLASS_NAMES):
    instance_img_path = os.path.join(scan_path, 'instance-filt', '{0}.png'.format(frame_name))
    if os.path.exists(instance_img_path):
        instance_img = np.asarray(Image.open(instance_img_path))
        instance_ids = np.unique(instance_img)
        for instance_id in instance_ids:
            # filter wrong instance_ids
            if instance_id not in objectID2label.keys(): continue
            if LABEL_MAP[objectID2label[instance_id]] in TARGET_CLASS_NAMES: return True
    return False


def find_valid_rotation_angle(current_framenames, scan_path, min_angle=5, to_visualize=False):
    selected_framenames = []

    while current_framenames != []:
        if len(current_framenames) == 1: break

        frame_name = current_framenames[0]
        src_pose = np.loadtxt(os.path.join(scan_path, 'pose', '{0}.txt'.format(frame_name)))

        num_candidates = len(current_framenames)
        for i in range(1, num_candidates):
            dst_pose = np.loadtxt(os.path.join(scan_path, 'pose', '{0}.txt'.format(current_framenames[i])))
            relative_angle = compute_relative_rotation(src_pose, dst_pose)
            if relative_angle >= min_angle:
                if to_visualize:
                    visualize_rotation_angle(scan_path, frame_name, current_framenames[i], relative_angle)
                selected_framenames.append(frame_name)
                current_framenames = current_framenames[i:]
                break
        else: break

    return selected_framenames


def main(opt):

    def log_string(out_str):
        LOG_FOUT.write(out_str + '\n')
        LOG_FOUT.flush()
        print(out_str)

    def select_frames_for_one_scan(scan_path, objectID2label):
        # Round1: check invalid pose file (discard those frames contain inf) e.g., scene0067_02/pose/133.txx
        num_frames = len(os.listdir(os.path.join(scan_path, 'color')))
        selected_framenames_r1 = []
        for i in range(num_frames):
            if find_valid_pose(scan_path, i):
                selected_framenames_r1.append(i)
        log_string('\tAfter filtering invalid pose files, remain {0} [origin {1}] frames'.format(
                    len(selected_framenames_r1), num_frames))

        # Round2: check the 2D segmentation annotation (discard those frames without target classes)
        selected_framenames_r2 = []
        for frame_name in selected_framenames_r1:
            if find_valid_objects(scan_path, frame_name, objectID2label, LABEL_MAP, TARGET_CLASS_NAMES):
                selected_framenames_r2.append(frame_name)
        log_string('\tAfter filtering frames without the interested objects, remain {0} [origin {1}] frames'.format(
                       len(selected_framenames_r2), len(selected_framenames_r1)))

        # Round3: check the camera rotation angle between two consecutive frames
        # (discard those frames with small view change)
        selected_framenames_r3 = find_valid_rotation_angle(selected_framenames_r2, scan_path,
                                                           opt.min_angle, opt.visu_validRot)
        log_string('\tAfter filtering frames with small viewpoint change, remain {0} [origin {1}] frames'.format(
                    len(selected_framenames_r3), len(selected_framenames_r2)))

        return selected_framenames_r3

    LOG_FOUT = open(os.path.join(opt.data_dir, 'log_process_data.txt'), 'w')
    LOG_FOUT.write(str(opt) + '\n')

    # map original class name into nyu40 class ids, and extract target classes
    LABEL_MAP_FILE = os.path.join(opt.data_dir, 'scannetv2-labels.combined.tsv')
    LABEL_MAP = scannet_utils.read_label_mapping(LABEL_MAP_FILE, label_from='raw_category', label_to='nyu40class')
    TARGET_CLASS_NAMES = cfg.SCANNET.CLASSES

    if opt.scene_name is None:
        SCAN_NAMES = [line.rstrip() for line in open('/mnt/Data/Datasets/ScanNet_v1/sceneid_sort.txt')]
        # import random
        # random.shuffle(SCAN_NAMES)
    else:
        SCAN_NAMES = [opt.scene_name]

    valid_scannames = []
    for scan_id, scan_name in enumerate(SCAN_NAMES):
        log_string('\n====== Process {0}-th scan [{1}] ======'.format(scan_id, scan_name))
        scan_path = os.path.join(opt.data_dir, 'scans', scan_name)

        if not os.path.exists(os.path.join(scan_path, 'instance-filt')): unzip_instace_file(scan_path, scan_name)
        objectID2label, colour_code = get_objectID2label_and_color(scan_path, scan_name)

        sel_framenames = select_frames_for_one_scan(scan_path, objectID2label)

        if len(sel_framenames) >= opt.min_frames:
            # During 2D bboxes extraction, the boxes whose dimension ratio to the image dimension is less than
            # min_ratio will be removed. This may lead to invalid without bbox..
            valid_framenames = extract_bbox2d_for_one_scan(scan_path, scan_name, sel_framenames, objectID2label,
                                                           LABEL_MAP, TARGET_CLASS_NAMES, colour_code,
                                                           opt.min_ratio, opt.visu_box2d, opt.save_box2d)
            log_string('\tAfter filtering objects by checking its dimension ratio to the image dimension, '
                                 'remain {0} [origin {1}] frames'.format(len(valid_framenames), len(sel_framenames)))
            if len(valid_framenames) > opt.min_frames:
                valid_scannames.append(scan_name)

    log_string('=======================================================')
    log_string('{0} scans are kept after processing...'.format(len(valid_scannames)))
    # save the valid scans..
    with open(os.path.join(opt.data_dir, 'sceneid_valid.txt'), 'w') as f:
        for scanname in valid_scannames:
            f.write('%s\n' %scanname)

    LOG_FOUT.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Select valid frames and extract 2D bboxes')
    parser.add_argument('--data_dir', type=str, default='/mnt/Data/Datasets/ScanNet_v2/',
                        help='The path to annotations')
    parser.add_argument('--scene_name', type=str, default=None, help='specific scene name to process')
    parser.add_argument('--min_frames', type=int, default=5, help='The minimum number of frames in one scene '
                                                                   'after removing invalid frames')
    parser.add_argument('--min_angle', type=float, default=5., help='The minimum rotation angle to filter frames')
    parser.add_argument('--min_ratio', type=int, default=15, help='The minimum ratio between the object dimension '
                                                                  'and the image dimension to filter objects')
    parser.add_argument('--visu_validRot', type=bool, default=True, help='Visualize valid rotation between '
                                                                            'two frames')
    parser.add_argument('--visu_box2d', type=bool, default=False, help='Visualize 2D bboxes')
    parser.add_argument('--save_box2d', type=bool, default=False, help='Save the 2D bbox into files..')

    opt = parser.parse_args()

    main(opt)

