import sys
# sys.path.append("../")
import os
import cv2
import numpy as np
import argparse
import torch
from torch.nn import functional as F
import torchvision.transforms as transforms

from config import cfg
import model
from data_loader import MyImageLoader

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default='/mnt/Data/Datasets/ScanNet_v2/', help='path to data')
parser.add_argument('--dataset', default='scannet', help='name of dataset')
parser.add_argument('--classes', type=int, default=13, help='number of target classes')
parser.add_argument('--input_size', type=int, default='484', help='the size of input images')
parser.add_argument('--model_name', type=str, default='ResNet50', help='name of model, options:[VGG19, ResNet50]')
parser.add_argument('--mode', type=str, default='train', help='specific dataset to process, options: [train, valid]')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints_20190918-20-24/ResNet50_11_0.71047.pth', #checkpoints_20190905-13-42/ResNet50_1_0.72084.pth
                                                   help='The path to the checkpoint')
parser.add_argument('--to_visualize', type=bool, default=True, help='Save the CAMs as heatmap images for visualization')
parser.add_argument('--to_save', type=bool, default=False, help='Save the CAMs as numpy array for further processing')
opt = parser.parse_args()

net = model.load_net(opt.model_name, opt.classes, opt.checkpoint_path).cuda()
finalconv_name = 'conv'

params = list(net.parameters())
# get weight only from the last layer(linear)
weight_softmax = params[-1].cpu().data

##========== Extract CAM for all classes with softmax normalization ==================#
def returnCAMs(feature_conv, weight_softmax, threshold=0.2):
    bz, nc, h, w = feature_conv.shape
    features =  feature_conv.reshape(nc, h*w).transpose(0,1) #(hxw, nc)
    CAMs = torch.mm(features, weight_softmax.transpose(0,1))  #(hxw, k)
    CAMs = CAMs - torch.min(CAMs, dim=0)[0]
    CAMs = CAMs / torch.max(CAMs, dim=0)[0]
    # threshold the corresponding CAM by 20% of its maximum value
    y = torch.zeros_like(CAMs)
    CAMs = torch.where(CAMs > threshold, CAMs, y)
    CAMs = CAMs.reshape(h, w, -1) #(h, w, k)
    return CAMs.numpy()


if __name__ == '__main__':

    dataset = MyImageLoader(opt.root_dir, opt.mode, opt.classes, opt.input_size)
    fileIds = dataset.scannet.all_valid_frames_list
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,  drop_last=False)

    if opt.to_visualize:
        # for visualization
        visu_dir = './result_image_{0}_ep{1}'.format(opt.checkpoint_path.split('/')[1][12:],
                                                     opt.checkpoint_path.split('/')[2].split('_')[1])
        os.makedirs(visu_dir, exist_ok=True)

    for i, (image_tensor, label_tensor) in enumerate(dataloader):
        scan_name = fileIds[i][0]
        frame_name = fileIds[i][1]
        sampleID = '{0}_{1}'.format(scan_name, frame_name)
        image_PIL = transforms.ToPILImage()(image_tensor[0])
        gt_label_vector = label_tensor.numpy()[0]
        print('Process sample: {0}'.format(sampleID))
        print('\tGround truth label is {0}'.format(gt_label_vector))

        image_tensor = image_tensor.cuda()
        logit, feature_blobs = net(image_tensor)
        feature_blobs = feature_blobs.cpu().data
        h_x = torch.sigmoid(logit).data.squeeze()
        print('\tPredicted class probability is [', ' '.join(f'{x:.4f}' for x in h_x.cpu().detach().numpy()), ']')

        CAMs = returnCAMs(feature_blobs, weight_softmax)

        if opt.to_save:
            save_dir = os.path.join(opt.root_dir, 'scans', scan_name, 'cam')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, '{0}.npy'.format(frame_name))
            np.save(save_path, CAMs)

        if opt.to_visualize:
            image_PIL.save(os.path.join(visu_dir, '{0}.jpg'.format(sampleID)))
            img = cv2.imread(os.path.join(visu_dir, '{0}.jpg'.format(sampleID)))
            height, width, _ = img.shape

            CAMs = np.uint8(255 * CAMs)

            # # save CAMs for all the classes
            # for idx in range(opt.classes):
            #     print('\t\tPredicted class index: {0}, class_name: {1}, probability: {2}'
            #                                 .format(idx, cfg.SCANNET.CLASSES[idx], h_x[idx].item()))
            #     CAM = CAMs[:,:,idx]
            #     heatmap = cv2.applyColorMap(cv2.resize(CAM, (width, height)), cv2.COLORMAP_JET)
            #     result = heatmap * 0.5 + img * 0.3
            #     cv2.imwrite(os.path.join(result_dir, '{0}_{1}_CAM.jpg'.format(sampleID, cfg.SCANNET.CLASSES[idx])), result)

            # only save the CAM with groud truth classes
            gt_label_indices = np.flatnonzero(gt_label_vector==1)
            for idx in gt_label_indices:
                print('\t\tPredicted class index: {0}, class_name: {1}, probability: {2}'
                                            .format(idx, cfg.SCANNET.CLASSES[idx], h_x[idx].item()))
                CAM = CAMs[:,:,idx]
                heatmap = cv2.applyColorMap(cv2.resize(CAM, (width, height)), cv2.COLORMAP_JET)
                result = heatmap * 0.5 + img * 0.3
                cv2.imwrite(os.path.join(visu_dir, '{0}_{1}_CAM.jpg'.format(sampleID, cfg.SCANNET.CLASSES[idx])), result)