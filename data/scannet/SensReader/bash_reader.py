#!/usr/bin/env python2
'''
Author: Zhao Na, 29 Aug 2019
'''
import os
import subprocess
from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser('Bash Read ScanNet')
parser.add_argument('--src_dir', type=str, default='/mnt/Data/Datasets/ScanNet_v1/scans/',
                    help='The directory storing source .sens files')
parser.add_argument('--des_dir', type=str, default='/mnt/Data/Datasets/ScanNet_v2/scans/',
                    help='The directory storing destinate color frames and camera parameters')
parser.add_argument('--scans_list_file', type=str, default='/mnt/Data/Datasets/ScanNet_v1/sceneid_sort.txt',
                    help='The path to the file storing the list of scan names to be read')

parser.add_argument('--export_depth_images', help='export_depth_images', action='store_true')
parser.add_argument('--export_color_images', help='export_color_images', action='store_true')
parser.add_argument('--export_poses', help='export_poses', action='store_true')
parser.add_argument('--export_intrinsics', help='export_intrinsics', action='store_true')

parser.add_argument('--num_processes', type=int, default=15, help='number of processes to parallelize')

opt = parser.parse_args()

def download(scan_name):
    print('====== Process scan [{0}] ======'.format(scan_name))
    filename = os.path.join(opt.src_dir, scan_name, '{0}.sens'.format(scan_name))
    output_path = os.path.join(opt.des_dir, scan_name)
    command = ["python2", "reader.py", "--filename", filename, "--output_path", output_path]
    if opt.export_depth_images:
        command.append("--export_depth_images")
    if opt.export_color_images:
        command.append("--export_color_images")
    if opt.export_poses:
        command.append("--export_poses")
    if opt.export_intrinsics:
        command.append("--export_intrinsics")
    subprocess.call(command)


if __name__ == '__main__':

    SCAN_NAMES = [line.rstrip() for line in open(opt.scans_list_file)]
    print('The number of scans to read is: {0}'.format(len(SCAN_NAMES)))

    pool = Pool(processes=opt.num_processes)
    pool.map(download, SCAN_NAMES)

    # for scan_id, scan_name in enumerate(SCAN_NAMES):
    #     print('====== Process {0}-th scan [{1}] ======'.format(scan_id, scan_name))
    #     filename = os.path.join(opt.src_dir, scan_name, '{0}.sens'.format(scan_name))
    #     output_path = os.path.join(opt.des_dir, scan_name)
    #     subprocess.call(["python2", "reader.py", "--filename", filename, "--output_path", output_path])