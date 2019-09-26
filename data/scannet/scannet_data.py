''' Help class and functions for loading ScanNet Objects
Author; Zhao Na
Date: Sep 2019
'''
import sys
import os
import cv2
import numpy as np
import pickle

import data.scannet.scannet_utils as utils
from config import cfg

class scannet_object(object):
    def __init__(self, root_dir, mode='train'):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, 'scans')
        self.mode = mode
        self.get_valid_scans()
        self.get_valid_frames()
        print('Mode-{0}: Scans-{1}, Frames-{2}'.format(self.mode, len(self.valid_scannames),
                                                       len(self.all_valid_frames_list)))

    def get_valid_scans(self):
        validIDs_file = os.path.join(self.root_dir, 'sceneid_valid.txt')
        all_validIDs = [x.strip() for x in open(validIDs_file).readlines()]

        split_file = os.path.join(self.root_dir, 'traintestsplit', 'scannetv2_{0}.txt'.format(self.mode))
        self.valid_scannames = []
        with open(split_file, 'r') as f:
            for line in f:
                scan_name = line.strip()
                if scan_name in all_validIDs:
                    self.valid_scannames.append(scan_name)

    def get_valid_frames(self):
        self.all_valid_frames_list = []
        for scan_name in self.valid_scannames:
            scan_dir = os.path.join(self.data_dir, scan_name)
            valid_frame_ids_file = os.path.join(scan_dir, '{0}_validframes_{1}class.txt'.format(
                                                            scan_name, len(cfg.SCANNET.CLASSES)))
            valid_frame_ids = [int(x.strip()) for x in open(valid_frame_ids_file).readlines()]
            for frame_id in valid_frame_ids:
                self.all_valid_frames_list.append((scan_name, frame_id))

    def get_image(self, scan_name, frame_id):
        img_file = os.path.join(self.data_dir, scan_name, 'color', '{0}.jpg'.format(frame_id))
        assert os.path.exists(img_file)
        return cv2.imread(img_file)

    def get_gt_2DBBox(self, scan_name, frame_id):
        label_file = os.path.join(self.data_dir, scan_name, 'bbox2d_{0}class'.format(len(cfg.SCANNET.CLASSES)),
                                                            '{0}_bbox.pkl'.format(frame_id))
        assert os.path.exists(label_file)
        return utils.read_2Dbbox(label_file)

    def __len__(self):
        return len(self.all_valid_frames_list)

    def __getitem__(self, index):
        scan_name, frame_id = self.all_valid_frames_list[index]
        img = self.get_image(scan_name, frame_id)
        bboxes2d = self.get_gt_2DBBox(scan_name, frame_id)
        return img, bboxes2d


def compute_overlap(objs):
    '''
    Input: objs, a list of n 2d objects,
                 each object is a dictionary {'classname': <str>, 'box2d': [minx,miny,maxx,maxy], 'instance_id':<int> }
    Output:
           classnames, numpy matrix, (n,)
           overlaps, numpy matrix, (n, n). where each element indicates the overlap between object i and object j.
    '''
    classnames = []
    bbox_dim = []
    for obj in objs:
        classnames.append(obj['classname'])
        bbox_dim.append(obj['box2d'])
    bbox_dim = np.array(bbox_dim) #(n,4)

    # Find intersections
    lower_bounds = np.maximum(np.expand_dims(bbox_dim[:, :2], axis=1), np.expand_dims(bbox_dim[:, :2], axis=0))  # (n, n, 2)
    upper_bounds = np.minimum(np.expand_dims(bbox_dim[:, 2:], axis=1), np.expand_dims(bbox_dim[:, 2:], axis=0))  # (n, n, 2)
    intersection_dims = np.clip(upper_bounds - lower_bounds, a_min=0, a_max=None)  # (n, n, 2)
    intersections = intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n, n)

    # Find areas of each box
    areas = (bbox_dim[:, 2] - bbox_dim[:, 0]) * (bbox_dim[:, 3] - bbox_dim[:, 1])  # (n)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    unions = np.expand_dims(areas, axis=1) + np.expand_dims(areas, axis=0) - intersections  # (n, n)

    overlaps = intersections/unions    # (n, n)

    return np.array(classnames), overlaps



def extract_rois(root_dir, mode, min_ratio=10, img_size=224):
    save_dir = os.path.join(root_dir, 'myscannet', 'RoIs')
    os.makedirs(save_dir, exist_ok=True)
    regions_indexing_list = []

    dataset = scannet_object(root_dir, mode)
    num_frames = len(dataset)
    print('Mode-{0}, Frames-{1}'.format(mode, num_frames))

    for idx in range(num_frames):
        scan_name, frame_idx = dataset.all_valid_frames_list[idx]
        img, bboxes2d = dataset.__getitem__(idx)
        classnames, overlaps = compute_overlap(bboxes2d)
        for bbox_idx, bbox in enumerate(bboxes2d):
            dimension = bbox['box2d']

            # assign labels to each region by considering labels of intersected bboxes...
            selected_idx = np.where(overlaps[bbox_idx, :] > 0)
            selected_classnames = classnames[selected_idx]
            multi_classname = list(set(selected_classnames))

            #TODO: augment data by box2d perturbation

            #filter out some invalid bbox (e.g., the width or height or area of bbox is lower than a threshold..)
            width = dimension[2]-dimension[0]
            height = dimension[3]-dimension[1]
            if width > (cfg.SCANNET.IMAGE_WIDTH/min_ratio) and height > (cfg.SCANNET.IMAGE_HEIGHT/min_ratio):
                regions_indexing_list.append({'scan_name': scan_name,
                                              'frame_idx': frame_idx,
                                              'bbox_idx': bbox_idx,
                                              'instance_id': bbox['instance_id'],
                                              'classname': bbox['classname'],
                                              'multi_classname': multi_classname,
                                              'dimension': dimension
                                              })

                roi = img[dimension[1]:dimension[3], dimension[0]:dimension[2]]
                cv2.imwrite(os.path.join(save_dir, '{0}-{1}-{2}.jpg'.format(scan_name, frame_idx, bbox_idx)), roi)

    print('Mode-{0}, Frames-{1}, RoIs-{2}'.format(mode, num_frames, len(regions_indexing_list)))
    with open(os.path.join(root_dir, 'myscannet', '{0}_roi_list.pkl'.format(mode)), 'wb') as f:
        pickle.dump(regions_indexing_list, f)



if __name__ == '__main__':
    root_dir = '/mnt/Data/Datasets/ScanNet_v2/'
    extract_rois(root_dir, mode='train')