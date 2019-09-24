import os

from config import cfg
from Box3dGenerator.tracker import tracking
from Box3dGenerator.frustums import compute_min_max_bounds_in_one_track


def process_one_scan(scan_dir, scan_name, valid_frame_names):
    objects, trajectories = tracking(scan_dir, scan_name, valid_frame_names, visu_pairwise_traj=False,
                                     visu_exhaustive_traj=False)

    # extract cams of objects
    for i, trajectory in enumerate(trajectories):
        print('Extract cams from the {0}-th trajectory [length: {1}]'.format(i, len(trajectory)))
        compute_min_max_bounds_in_one_track(scan_dir, objects, trajectory)


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
        import random
        random.shuffle(scan_name_list)

    for scan_idx, scan_name in enumerate(scan_name_list):
        print('-----------Process ({0}, {1})-----------'.format(scan_idx, scan_name))
        scan_dir = os.path.join(data_dir, scan_name)
        valid_frame_names_file = os.path.join(scan_dir, '{0}_validframes_18class_{1}frameskip.txt'
                                            .format(scan_name, cfg.SCANNET.FRAME_SKIP))
        valid_frame_names = [int(x.strip()) for x in open(valid_frame_names_file).readlines()]

        process_one_scan(scan_dir, scan_name, valid_frame_names)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/mnt/Data/Datasets/ScanNet_v2/', help='path to data')
    parser.add_argument('--scan_id', default='scene0067_02', help='specific scan id to download') #'scene0067_02'

    opt = parser.parse_args()

    main(opt)