#!/usr/bin/env python2
'''
Original from https://github.com/ScanNet/ScanNet
Modified by Zhao Na, 28 Aug 2019
'''

import argparse
import os, sys

from SensorData import SensorData

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--filename', required=True, help='path to sens file to read')
parser.add_argument('--output_path', required=True, help='path to output folder')
parser.add_argument('--export_depth_images', help='export_depth_images', action='store_true')
parser.add_argument('--export_color_images', help='export_color_images', action='store_true')
parser.add_argument('--export_poses', help='export_poses', action='store_true')
parser.add_argument('--export_intrinsics', help='export_intrinsics', action='store_true')

opt = parser.parse_args()
print('-----------------------------')
print(opt)
print('-----------------------------')


def main():
  if not os.path.exists(opt.output_path):
    os.makedirs(opt.output_path)
  # load the data
  sys.stdout.write('loading %s...' % opt.filename)
  sd = SensorData(opt.filename)
  sys.stdout.write('loaded!\n')
  if opt.export_depth_images:
    sd.export_depth_images(os.path.join(opt.output_path, 'depth'), frame_skip=1)
  if opt.export_color_images:
    sd.export_color_images(os.path.join(opt.output_path, 'color'), frame_skip=1)
  if opt.export_poses:
    sd.export_poses(os.path.join(opt.output_path, 'pose'), frame_skip=1)
  if opt.export_intrinsics:
    sd.export_intrinsics(os.path.join(opt.output_path, 'intrinsic'))


if __name__ == '__main__':
    main()