import sys
sys.path.append("../")
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from sklearn.metrics import average_precision_score
import argparse
import ast
from datetime import datetime
from collections import Counter

import torch
from tensorboardX import SummaryWriter

from config import cfg
import model
from focalloss import FocalLoss2
from data_loader import MyImageLoader

parser = argparse.ArgumentParser('Train multi-class classification model for extracting CAM')
parser.add_argument('--gpu_ids', default='[2]', help='The GPUs to do data parallel')
parser.add_argument('--root_dir', default='/mnt/Data/Datasets/ScanNet_v2/', help='path to data')
parser.add_argument('--dataset', default='scannet', help='name of dataset')
parser.add_argument('--classes', type=int, default=13, help='number of target classes')
parser.add_argument('--input_size', type=int, default='484', help='the size of input images') #484, rescale the image into half

parser.add_argument('--model_name', type=str, default='ResNet50', help='name of model, options:[VGG19, ResNet50]')
parser.add_argument('--pretrained', action='store_true', help='If pre-train the model on ImageNet')
parser.add_argument('--mAP_threshold', type=float, default=0.7, help='The mAP threshold ot save the checkpoint')
parser.add_argument('--focalloss_gamma', type=float, default=0, help='Whether use focal loss (0: not use; >0 use, '
                                                         'and the value is the setting of the gamma) [options:0.0-5.0]')
parser.add_argument('--use_posweight', action='store_true', help='use pos_weight for loss function')

parser.add_argument('--checkpoint_path', type=str, default=None, help='The path to the checkpoint')
parser.add_argument('--nThreads', default=10, type=int, help='# threads for loading data')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training')
parser.add_argument('--max_epoch', type=int, default=20, help='Epoch to run [default: 100]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay value for optimizer [default: 0]')
parser.add_argument('--decay_step', type=int, default=5, help='Decay step (epoch) for lr decay [default: 20]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.5]')
opt = parser.parse_args()

args = vars(opt)
print('------------------ Options ------------------')
for k, v in sorted(args.items()):
    print('{0}:{1}'.format(k,v))
print('Class2Index:\n {0}'.format(cfg.SCANNET.CLASS2INDEX))
print('-------------------- End --------------------')

GPU_ID = ast.literal_eval(opt.gpu_ids)


def evaluate(pred, gt, n_class):
    '''
    :param pred: numpy array (n_samples, n_class)
    :param gt:  numpy array (n_samples, n_class)
    :param n_class: scalar, the number of classes
    :return: mAP
    '''
    assert n_class == pred.shape[1] == gt.shape[1]
    ap = []
    for c in range(n_class):
        gt_c = gt[:,c]
        pred_c = pred[:,c]
        ap_c = average_precision_score(gt_c, pred_c)
        ap.append(ap_c)
    print('\tAP for all the classes:\n \t{0}'.format(ap))
    mAP = np.mean(np.array(ap))
    return ap, mAP


def calculate_pos_weight(num_class, image_labels):
        '''
        compute the number the samples for each class across the dataset, and calculate pos_weight for loss function
            For example, if a dataset contains 100 positive and 300 negative examples of a single class,
            then pos_weight for the class should be equal to 300/100=3. The loss would act as if the
            dataset contains 3×100=300 positive examples.
        '''
        class_distrib = Counter([c for img in image_labels for c in img])
        print('\tClass distribution: ', class_distrib)
        assert len(class_distrib) == num_class
        dataset_size = len(image_labels)

        pos_weight = np.zeros(num_class)
        for class_name, num_samples in class_distrib.items():
            pos_weight[cfg.SCANNET.CLASS2INDEX[class_name]] = (dataset_size - num_samples) / num_samples
        print('\tPos_weight: ', pos_weight)

        return pos_weight


if __name__ == '__main__':

    print('Loading data...')
    trainset = MyImageLoader(opt.root_dir, 'train', opt.classes, opt.input_size)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True,
                                              num_workers=opt.nThreads,  drop_last=True)
    validset = MyImageLoader(opt.root_dir, 'valid', opt.classes, opt.input_size)
    validloader = torch.utils.data.DataLoader(validset, batch_size=opt.batch_size, shuffle=False,
                                              num_workers=opt.nThreads,  drop_last=True)

    if opt.checkpoint_path is not None:
        net = model.load_net(opt.classes, opt.checkpoint_path).cuda()
    else:
        net = model.get_net(opt.model_name, opt.classes, opt.pretrained).cuda()

    if len(GPU_ID) > 1:
        net = torch.nn.DataParallel(net, device_ids=GPU_ID)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID[0])

    # set loss function
    if opt.use_posweight:
        print('Input pos_weight into loss function...')
        pos_weight = calculate_pos_weight(opt.classes, trainset.image_classes)
        pos_weight = torch.from_numpy(pos_weight.astype(np.float32)).cuda()
    else:
        pos_weight = None

    if opt.focalloss_gamma > 0:
        criterion = FocalLoss2(gamma=opt.focalloss_gamma, pos_weight=pos_weight).cuda()
    elif opt.focalloss_gamma == 0:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).cuda()
    else:
        raise Exception('The gamma in FocalLoss must be non-negative!')

    cur_lr = opt.learning_rate
    optimizer = torch.optim.Adam(net.parameters(), lr=cur_lr, weight_decay=opt.weight_decay)

    writer = SummaryWriter()
    # write log information
    writer.add_text('config', 'Dataset: %s' % opt.dataset, 0)
    writer.add_text('config', 'Model Name: %s' % opt.model_name, 0)
    writer.add_text('config', 'Model Pretained: %s' % str(opt.pretrained), 0)
    writer.add_text('config', 'Resumed Checkpoint: %s' % opt.checkpoint_path, 0)
    writer.add_text('config', 'Num classes: %d' % opt.classes, 0)
    writer.add_text('config', 'Input Image Size: %d' % opt.input_size, 0)
    writer.add_text('config', 'FocalLoss2 Gamma: %f' % opt.focalloss_gamma, 0)
    writer.add_text('config', 'Use Pos_weight: %s' % str(opt.use_posweight), 0)
    writer.add_text('config', 'Maximum Epoch: %d' % opt.max_epoch, 0)
    writer.add_text('config', 'Batch Size: %d' % opt.batch_size, 0)
    writer.add_text('config', 'Learning Rate: %.5f' %opt.learning_rate, 0)
    writer.add_text('config', 'Weight Decay: %.5f' %opt.weight_decay)
    writer.add_text('config', 'Decay Step: %d' %opt.decay_step, 0)
    writer.add_text('config', 'Decay Rate: %.2f' %opt.decay_rate, 0)

    save_model_dir = './checkpoints_' + datetime.now().strftime('%Y%m%d-%H-%M')
    if not os.path.exists(save_model_dir): os.mkdir(save_model_dir)

    print("START TRAINING")
    n_iter = 0
    best_mAP = 0
    for epoch in range(opt.max_epoch):
        epoch_loss = 0
        for i, (images, labels) in enumerate(trainloader):
            n_iter += 1
            images, labels = images.cuda(), labels.cuda()
            net.train()
            optimizer.zero_grad()
            outputs, _ = net(images)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            writer.add_scalar('train/loss', loss, n_iter)

            if (i+1) % 200 == 0:
                print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f'
                      % (epoch+1, opt.max_epoch, i+1, len(trainloader), loss.item()))

        avg_epoch_loss = epoch_loss / len(trainloader)
        print("Epoch: %d, Avg Loss: %.4f" % (epoch+1, avg_epoch_loss))

        # evaluate on validation datset
        if epoch % 1 == 0:
            print('Evaluating...')
            with torch.no_grad():
                net.eval()
                gt_label = []
                pred_label = []
                test_loss = 0
                num_samples = 0
                for i, (images, labels) in enumerate(validloader):
                    images, labels = images.cuda(), labels.cuda()
                    outputs, _ = net(images)
                    loss = criterion(outputs, labels)

                    test_batch_size = labels.shape[0]
                    test_loss += loss.detach() * test_batch_size
                    num_samples += test_batch_size

                    pred = torch.sigmoid(outputs)
                    gt_label.append(labels.cpu().detach())
                    pred_label.append(pred.cpu().detach())

                test_loss /= num_samples
                writer.add_scalar('valid/loss', test_loss, n_iter)

                gt_label = torch.stack(gt_label, dim=0).view(-1, opt.classes).numpy()
                pred_label = torch.stack(pred_label, dim=0).view(-1, opt.classes).numpy()
                ap, mAP = evaluate(pred_label, gt_label, opt.classes)
                print('\tTest network. The mAP of the {0}-th epoch is {1}'.format(epoch+1, mAP))
                writer.add_scalar('valid/mAP', mAP, n_iter)
                writer.add_text('AP on valid', '{}'.format(ap), epoch+1)

                if mAP > opt.mAP_threshold:
                    model.save_net(net, save_model_dir, '%s_%d_%.5f' %(opt.model_name, epoch+1, mAP), GPU_ID)
                    print('\t\tmAP > {0}, saving model...'.format(opt.mAP_threshold))

        # reduce learning rate
        if (epoch+1) % opt.decay_step == 0:
            cur_lr = model.update_learning_rate(cur_lr, opt.decay_rate, optimizer)
        writer.add_scalar('train/learning_rate', cur_lr, n_iter)

        print("----------------------------------")

    writer.close()