import sys
sys.path.append("../")
import os
import cv2
import numpy as np
import argparse
import torch
from torch.nn import functional as F
import torchvision.transforms as transforms

from config import cfg
import model
from data_loader import MyRoILoader, MyImageLoader

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default='/mnt/Data/Datasets/ScanNet_v2/', help='path to data')
parser.add_argument('--dataset', default='scannet', help='name of dataset')
parser.add_argument('--classes', type=int, default=18, help='number of target classes')
parser.add_argument('--input_size', type=int, default='224', help='the size of input images')
parser.add_argument('--model_name', type=str, default='ResNet50', help='name of model, options:[VGG19, ResNet50]')
parser.add_argument('--input_type', type=str, default='roi', help='the type of input data, options: [roi, image]. '
                    'If roi is selected, MyRoILoader will be used, elif image is selected, MyImageLoader will be used')
parser.add_argument('--mode', type=str, default='train', help='specific dataset to process, options: [train, valid]')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints_20190905-13-25/ResNet50_2_0.61141.pth',
                                                   help='The path to the checkpoint')
opt = parser.parse_args()

result_dir = './result_{0}_{1}_ep{2}'.format(opt.input_type, opt.checkpoint_path.split('/')[1][12:],
                                             opt.checkpoint_path.split('/')[2].split('_')[1])
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

if opt.input_type == 'image':
    dataset = MyImageLoader(opt.root_dir, opt.mode, opt.classes, opt.input_size)
    fileIds = ['{0}_{1}'.format(scan_name, frame_id) for (scan_name, frame_id) in dataset.scannet.all_valid_frames_list]
elif opt.input_type == 'roi':
    dataset = MyRoILoader(opt.root_dir, opt.mode, opt.classes, opt.input_size)
    fileIds = ['{0}_{1}_{2}'.format(roi['scan_name'], roi['frame_idx'], roi['bbox_idx']) for roi in dataset.roi_list]
else:
    raise ValueError('Error! The input_type argument can only be set as roi or image')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,  drop_last=False)

net = model.load_net(opt.model_name, opt.classes, opt.checkpoint_path).cuda()
finalconv_name = 'conv'

params = list(net.parameters())
# get weight only from the last layer(linear)
weight_softmax = params[-1].cpu().data

# #========== Extract CAM only for high-probability predicted classes ==================

# # hook
# feature_blobs = []
# def hook_feature(module, input, output):
#     feature_blobs.append(output.cpu().data.numpy())
#
# net._modules.get(finalconv_name).register_forward_hook(hook_feature)
#
# params = list(net.parameters())
# # get weight only from the last layer(linear)
# weight_softmax = np.squeeze(params[-1].cpu().data.numpy())

# def returnCAM(feature_conv, weight_softmax, class_idx):
#     size_upsample = (128, 128)
#     bz, nc, h, w = feature_conv.shape
#     output_cam = []
#     cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
#     cam = cam.reshape(h, w)
#     cam = cam - np.min(cam)
#     cam_img = cam / np.max(cam)
#     cam_img = np.uint8(255 * cam_img)
#     output_cam.append(cv2.resize(cam_img, size_upsample))
#     return output_cam
#
# for i, (image_tensor, label_tensor) in enumerate(validloader):
#     sampleID = valid_fileIds[i]
#     print('\tProcess sample: {0}'.format(sampleID))
#     print('\t\tGround truth label is {0}'.format(label_tensor.numpy()[0]))
#     image_PIL = transforms.ToPILImage()(image_tensor[0])
#
#     image_tensor = image_tensor.cuda()
#     logit, _ = net(image_tensor)
#     h_x = torch.sigmoid(logit).data.squeeze()
#     print('\t\tPredicted class probability is [', ' '.join(f'{x:.4f}' for x in h_x.cpu().detach().numpy()), ']')
#     pred_class_idx = (h_x>0.5).nonzero()
#     if len(pred_class_idx) != 0:
#         image_PIL.save('result/%06d.jpg' % sampleID)
#         for idx in pred_class_idx:
#             idx = idx.item()
#             print('\t\tPredicted class index: {0}, class_name: {1}, probability: {2}'
#                         .format(idx, classes[idx], h_x[idx].item()))
#             CAM = returnCAM(feature_blobs[0], weight_softmax, idx)
#             img = cv2.imread('result/%06d.jpg' %sampleID)
#             height, width, _ = img.shape
#             heatmap = cv2.applyColorMap(cv2.resize(CAM[0], (width, height)), cv2.COLORMAP_JET)
#             result = heatmap * 0.5 + img * 0.5
#             cv2.imwrite('result/%06d_%s_CAM.jpg' %(sampleID, classes[idx]), result)
#     else:
#         print('\t\tNone of predicted class probability is larger than 0.5.')



##========== Extract CAM for all classes with softmax normalization ==================#
def returnCAMs(feature_conv, weight_softmax):
    bz, nc, h, w = feature_conv.shape
    features =  feature_conv.reshape(nc, h*w).transpose(0,1) #(hxw, nc)
    CAMs = torch.mm(features, weight_softmax.transpose(0,1))  #(hxw, k)
    #softmax normalization along K classes
    CAMs = F.softmax(CAMs, dim=1)
    CAMs = CAMs.reshape(h, w, -1) #(h, w, k)
    CAMs = np.uint8(255 * CAMs.numpy())
    return CAMs


for i, (image_tensor, label_tensor) in enumerate(dataloader):
    sampleID = fileIds[i]
    gt_label_vector = label_tensor.numpy()[0]
    print('Process sample: {0}'.format(sampleID))
    print('\tGround truth label is {0}'.format(gt_label_vector))
    image_PIL = transforms.ToPILImage()(image_tensor[0])
    image_PIL.save(os.path.join(result_dir, '{0}.jpg'.format(sampleID)))

    image_tensor = image_tensor.cuda()
    logit, feature_blobs = net(image_tensor)
    feature_blobs = feature_blobs.cpu().data
    h_x = torch.sigmoid(logit).data.squeeze()
    print('\tPredicted class probability is [', ' '.join(f'{x:.4f}' for x in h_x.cpu().detach().numpy()), ']')

    CAMs = returnCAMs(feature_blobs, weight_softmax)
    img = cv2.imread(os.path.join(result_dir, '{0}.jpg'.format(sampleID)))
    height, width, _ = img.shape
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
        cv2.imwrite(os.path.join(result_dir, '{0}_{1}_CAM.jpg'.format(sampleID, cfg.SCANNET.CLASSES[idx])), result)

