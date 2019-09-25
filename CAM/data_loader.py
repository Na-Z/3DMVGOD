import os
import numpy as np
import pickle
from PIL import Image, ImageOps
import torch
import torch.utils.data as DATA
import torchvision.transforms.functional as F

from data.scannet.scannet_data import scannet_object
from config import cfg

class MyRoILoader(DATA.Dataset):
    def __init__(self, root_dir, mode, num_class, input_size):
        super(MyRoILoader, self).__init__()
        self.data_dir = os.path.join(root_dir, 'myscannet', 'RoIs')
        self.num_class = num_class
        self.input_size = input_size

        roi_list_file = os.path.join(root_dir, 'myscannet', '{0}_roi_list.pkl'.format(mode))
        with open(roi_list_file, 'rb') as f:
            self.roi_list = pickle.load(f)
        print('Mode-{0}: RoIs-{1}'.format(mode, len(self.roi_list)))

    def __len__(self):
        return len(self.roi_list)

    def __getitem__(self, index):
        roi = self.roi_list[index]
        img_path = os.path.join(self.data_dir, '{0}-{1}-{2}.jpg'.format(roi['scan_name'], roi['frame_idx'], roi['bbox_idx']))

        img = Image.open(img_path)
        w, h = img.size
        delta_w = cfg.SCANNET.IMAGE_WIDTH - w
        delta_h = cfg.SCANNET.IMAGE_HEIGHT - h
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        new_img = ImageOps.expand(img, padding)
        # resize the image into a low resolution
        new_img = F.resize(new_img, self.input_size, interpolation=Image.LANCZOS)
        # new_img.show()

        label = np.zeros(self.num_class)
        for class_name in roi['multi_classname']:
            label[cfg.SCANNET.CLASS2INDEX[class_name]] = 1

        image = F.to_tensor(new_img)
        label = torch.from_numpy(label.astype(np.float32))

        return image, label


class MyImageLoader(DATA.Dataset):
    def __init__(self, root_dir, mode, num_class, input_size):
        super(MyImageLoader, self).__init__()
        self.num_class = num_class
        assert num_class == len(cfg.SCANNET.CLASSES)
        self.input_size = input_size

        self.scannet = scannet_object(root_dir, mode)
        self.extract_image_info()
        print('Mode-{0}: Images-{1}'.format(mode, len(self.image_paths)))

    def extract_image_info(self):
        '''
        Extract path and labels for each image
        '''
        self.image_paths = []
        self.image_classes = []
        for idx, (scan_name, frame_id) in enumerate(self.scannet.all_valid_frames_list):
            img_path = os.path.join(self.scannet.data_dir, scan_name, 'color', '{0}.jpg'.format(frame_id))
            objs = self.scannet.get_gt_2DBBox(scan_name, frame_id)
            classname_list = set([obj['classname'] for obj in objs])
            self.image_paths.append(img_path)
            self.image_classes.append(list(classname_list))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        class_names = self.image_classes[index]
        label = np.zeros(self.num_class)
        for class_name in class_names:
            label[cfg.SCANNET.CLASS2INDEX[class_name]] = 1

        img = Image.open(self.image_paths[index])
        ## resize the image into a low resolution
        new_img = F.resize(img, self.input_size, interpolation=Image.LANCZOS)
        image = F.to_tensor(new_img)
        label = torch.from_numpy(label.astype(np.float32))

        return image, label