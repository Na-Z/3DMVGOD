import os

from config import cfg
from Box3dGenerator.tracker import tracking
from Box3dGenerator.frustums import compute_min_max_bounds_in_one_track


def process_one_scan(scan_dir, scan_name, valid_frame_names):
    #TODO: input for the arguments of tracking function
    objects, trajectories = tracking(scan_dir, scan_name, valid_frame_names, min_ratio=9,
                                     pairwise_min_scale=0.6, pairwise_min_dist=50,
                                     visu_pairwise_epipolar=False, visu_pairwise_traj=False,
                                     exhaustive_min_dist=100, exhaustive_max_pair_angle=110,
                                     exhausive_min_size_ratio=0.7, min_track_length=4,
                                     min_coverage_angle=20, visu_exhaustive_epipolar=False,
                                     visu_exhaustive_rotation=False, visu_exhaustive_traj=True)

    # extract cams of objects
    for i, trajectory in enumerate(trajectories):
        print('Generate 3D bbox for the {0}-th trajectory [length: {1}]'.format(i, len(trajectory)))
        # if i <= 0: continue
        compute_min_max_bounds_in_one_track(scan_dir, scan_name, objects, trajectory)


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
        print('------------------Process ({0}, {1})------------------'.format(scan_idx, scan_name))
        scan_dir = os.path.join(data_dir, scan_name)
        # TODO: modify file name, remove {2}frameskip
        # valid_frame_names_file = os.path.join(scan_dir, '{0}_validframes_{1}class.txt'
        #                                       .format(scan_name, len(cfg.SCANNET.CLASSES)))
        valid_frame_names_file = os.path.join(scan_dir, '{0}_validframes_18class_{1}frameskip.txt'
                                                        .format(scan_name, cfg.SCANNET.FRAME_SKIP))
        valid_frame_names = [int(x.strip()) for x in open(valid_frame_names_file).readlines()]

        process_one_scan(scan_dir, scan_name, valid_frame_names)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/mnt/Data/Datasets/ScanNet_v2/', help='path to data')
    parser.add_argument('--scan_id', default='scene0431_00', help='specific scan id to download') #'scene0067_02','scene0543_02','scene0299_00','scene0431_00'

    opt = parser.parse_args()

    main(opt)