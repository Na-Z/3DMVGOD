#!/usr/bin/env python2
'''
Author: Zhao Na, 29 Aug 2019
'''
import os
import subprocess

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Bash Read ScanNet with Frame Skip')
    parser.add_argument('--src_dir', type=str, default='/mnt/Data/Datasets/ScanNet_v1/scans/',
                        help='The directory storing source .sens files')
    parser.add_argument('--des_dir', type=str, default='/mnt/Data/Datasets/ScanNet_v2/scans/',
                        help='The directory storing destinate color frames and camera parameters')
    parser.add_argument('--scans_list_file', type=str, default='/mnt/Data/Datasets/ScanNet_v1/sceneid_sort.txt',
                        help='The path to the file stoing the list of scan names to be read')

    opt = parser.parse_args()

    SCAN_NAMES = [line.rstrip() for line in open(opt.scans_list_file)]

    for scan_id, scan_name in enumerate(SCAN_NAMES):
        print('====== Process {0}-th scan [{1}] ======'.format(scan_id, scan_name))
        filename = os.path.join(opt.src_dir, scan_name, '{0}.sens'.format(scan_name))
        output_path = os.path.join(opt.des_dir, scan_name)
        subprocess.call(["python2", "reader.py", "--filename", filename, "--output_path", output_path])