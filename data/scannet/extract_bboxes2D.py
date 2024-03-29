import os
import numpy as np
import cv2
from PIL import Image
from zipfile import ZipFile
from vtk_visualizer.plot3d import *

import scannet_utils
from config import cfg

OBJECT_LIST = []


def unzip_instace_file(scan_path, scan_name):
    ##  unzip the folder storing instance segmentation annotations #####
    ##  <scanId>_2d-instance-filt.zip (Filtered 2d projections of aggregated annotation instances as 8-bit pngs)
    instace_file = os.path.join(scan_path, '{0}_2d-instance-filt.zip'.format(scan_name))
    with ZipFile(instace_file, 'r') as zip_ref:
        zip_ref.extractall(scan_path)


def generate_colors(n):
    r = int(random.random()*256)
    g = int(random.random()*256)
    b = int(random.random()*256)
    step = 256 / n
    r_list = []
    g_list = []
    b_list = []
    for idx in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        r_list.append(r)
        g_list.append(g)
        b_list.append(b)
    random.shuffle(r_list)
    random.shuffle(g_list)
    random.shuffle(b_list)

    rgb_values = np.vstack((np.array(r_list), np.array(g_list), np.array(b_list)))
    return np.transpose(rgb_values)


def get_objectID2label_and_color(scan_path, scan_name):
    # parse the annotation json file
    agg_file = os.path.join(scan_path, '{0}_vh_clean.aggregation.json'.format(scan_name))
    objectID2label = scannet_utils.read_aggregation(agg_file)
    colour_code = generate_colors(len(objectID2label))
    return objectID2label, colour_code


def compute_box2d(x_indices, y_indices):
    return [np.min(x_indices), np.min(y_indices), np.max(x_indices), np.max(y_indices)]


def extract_bbox2d_from_instance(instance_img_path, objectID2label, LABEL_MAP, TARGET_CLASS_NAMES,
                                 colour_code, min_ratio=10, visualize=True):
    instance_img = np.asarray(Image.open(instance_img_path))
    h, w = instance_img.shape
    class_img_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    r = class_img_rgb[:, :, 0]
    g = class_img_rgb[:, :, 1]
    b = class_img_rgb[:, :, 2]

    instance_ids = np.unique(instance_img)
    objects2d = []
    for instance_id in instance_ids:
        # filter wrong instance_ids
        if instance_id not in objectID2label.keys(): continue

        classname_original = objectID2label[instance_id]
        classname_nyu40id = LABEL_MAP[classname_original]

        if classname_nyu40id in TARGET_CLASS_NAMES:
            y_indices, x_indices = np.where(instance_img==instance_id)
            box2d = compute_box2d(x_indices, y_indices)

            # filter out some invalid bbox (e.g., the width or height or area of bbox is lower than a threshold..)
            width = box2d[2] - box2d[0]
            height = box2d[3] - box2d[1]
            if width > (cfg.SCANNET.IMAGE_WIDTH / min_ratio) and height > (cfg.SCANNET.IMAGE_HEIGHT / min_ratio):
                objects2d.append({'box2d': box2d, 'classname': classname_nyu40id, 'instance_id': instance_id})

        r[instance_img==instance_id] = np.uint8(colour_code[instance_id-1][0])
        g[instance_img==instance_id] = np.uint8(colour_code[instance_id-1][1])
        b[instance_img==instance_id] = np.uint8(colour_code[instance_id-1][2])

    class_img_rgb[:,:,0] = r
    class_img_rgb[:,:,1] = g
    class_img_rgb[:,:,2] = b

    if visualize and len(objects2d) > 0:
        ## visualize the semantic labeled frame
        # Image.fromarray(class_img_rgb).show()

        rgb_img = instance_img_path.replace('instance-filt', 'color').replace('png', 'jpg')
        rgb_img = cv2.imread(rgb_img)
        for obj in objects2d:
            # in case the xmin/ymin or xmax/ymax is at the border of the image..
            cv2.rectangle(rgb_img, (max(int(obj['box2d'][0]), 2), max(int(obj['box2d'][1]), 2)),
                                   (min(int(obj['box2d'][2]), w-2), min(int(obj['box2d'][3]), h-2)),
                                   (0, 255, 0), 2)
            cv2.putText(rgb_img, '%d %s' % (obj['instance_id'], obj['classname']),
                        (max(int(obj['box2d'][0]), 15), max(int(obj['box2d'][1]), 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        # Image.fromarray(rgb_img).show()
        cv2.imshow('2dbox', rgb_img)
        cv2.waitKey(0)

    return objects2d


def extract_bbox2d_for_one_scan(scan_path, scan_name, frame_names, objectID2label, LABEL_MAP, TARGET_CLASS_NAMES,
                                colour_code, min_ratio, to_visu, to_save=False):
    if to_save:
        bbox_dir = os.path.join(scan_path, 'bbox2d_{0}class'.format(len(TARGET_CLASS_NAMES)))
        if not os.path.exists(bbox_dir): os.mkdir(bbox_dir)

    valid_framenames = []
    for frame_name in frame_names:
        instance_img_path = os.path.join(scan_path, 'instance-filt', '{0}.png'.format(frame_name))
        objects2d = extract_bbox2d_from_instance(instance_img_path, objectID2label, LABEL_MAP, TARGET_CLASS_NAMES,
                                                 colour_code,  min_ratio, visualize=to_visu)
        if len(objects2d) > 0:
            valid_framenames.append(frame_name)
            if to_save:
                # save 2D bboxes into file
                scannet_utils.write_2Dbbox(bbox_dir, frame_name, objects2d)

    if to_save:
        save_path = os.path.join(scan_path, '{0}_validframes_{1}class.txt'. format(scan_name, len(TARGET_CLASS_NAMES)))
        with open(save_path, 'w') as f:
            for frame_name in valid_framenames:
                f.write('%d\n' % frame_name)

    return valid_framenames



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Generate 2D bbox GroundTruth from frame-level instance segmentation annotation')
    parser.add_argument('--data_dir', type=str, default='/mnt/Data/Datasets/ScanNet_v2/scans/',
                        help='The path to annotations')
    parser.add_argument('--frame_skip', type=int, default=15,
                        help='the number of frames to skip in extracting instance annotation images')
    parser.add_argument('--scene_name', type=str, default=None, help='specific scene name to process')
    parser.add_argument('--min_ratio', type=int, default=10, help='The minimum ratio between the object dimension '
                                                                  'and the image dimension to filter objects')
    parser.add_argument('--to_visu', type=bool, default=True, help='Visualize 2D bboxes')
    parser.add_argument('--to_save', type=bool, default=True, help='Save the 2D bbox into files..')

    opt = parser.parse_args()

    # map original class name into nyu40 class ids, and extract 18 target classes
    LABEL_MAP_FILE = '/mnt/Data/Datasets/ScanNet_v2/scannetv2-labels.combined.tsv'
    LABEL_MAP = scannet_utils.read_label_mapping(LABEL_MAP_FILE, label_from='raw_category', label_to='nyu40class')
    TARGET_CLASS_NAMES = cfg.SCANNET.CLASSES

    if opt.scene_name is None:
        SCAN_NAMES = [line.rstrip() for line in open('/mnt/Data/Datasets/ScanNet_v1/sceneid_sort.txt')]
    else:
        SCAN_NAMES = [opt.scene_name]

    for scan_id, scan_name in enumerate(SCAN_NAMES):
        print('====== Process {0}-th scan [{1}] ======'.format(scan_id, scan_name))
        scan_path = os.path.join(opt.data_dir, scan_name)

        if not os.path.exists(os.path.join(scan_path, 'instance-filt')): unzip_instace_file(scan_path, scan_name)
        objectID2label, colour_code = get_objectID2label_and_color(scan_path, scan_name)

        frame_names = os.listdir(os.path.join(scan_path, 'color'))
        valid_framenames = extract_bbox2d_for_one_scan(scan_path, scan_name, frame_names, objectID2label, LABEL_MAP, TARGET_CLASS_NAMES,
                                    colour_code, opt.min_ratio, opt.to_visu, opt.to_save)
