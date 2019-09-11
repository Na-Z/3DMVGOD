"""
This file is for model implementation.
It has Convolution layers and Average pooling.
"""
import os
import torch
import torch.nn as nn
from torchvision import models as MODELZOO

class VGG19(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(VGG19, self).__init__()
        model = MODELZOO.vgg19(pretrained=pretrained)
        features_module = list(model.features.children())
        # remove pool5
        features_module.pop()
        # 512 x 14 x 14 (if input size is 224*224)
        # add one convolutional layer of size 3×3, stride 1, pad 1 with 1024 units, before the GAP layer (followed Zhou CVPR2016)
        features_module.append(nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1))
        self.conv = nn.Sequential(*features_module)
        # 1024 x 14 x 14 (if input size is 224*224)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # 1024 x 1 x 1 (if input size is 224*224)
        self.classifier = nn.Linear(1024, num_classes, bias=False) #bias=False

    def forward(self, x):
        features = self.conv(x)
        flatten = self.avg_pool(features).view(features.size(0), -1)
        output = self.classifier(flatten)
        return output, features


class ResNet50(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(ResNet50, self).__init__()
        model = MODELZOO.resnet50(pretrained=pretrained)
        original_module = list(model.children())
        # remove last linear layer and avgpooling layer
        features_module = original_module[:-2]
        self.conv = nn.Sequential(*features_module)
        # 2048 x m x m
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # 2048 x 1 x 1
        self.classifier = nn.Linear(2048, num_classes, bias=False) #bias=False

    def forward(self, x):
        features = self.conv(x)
        flatten = self.avg_pool(features).view(features.size(0), -1)
        output = self.classifier(flatten)
        return output, features


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_net(model_name, num_classes, pretrained):
    Model = globals()[model_name]
    net = Model(num_classes, pretrained)
    # net.apply(weight_init)
    print("INIT NETWORK")
    return net


def load_net(model_name, num_classes, checkpoint_path):
    # net = CNN(num_classes)
    Model = globals()[model_name]
    net = Model(num_classes)
    net.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    return net


def save_net(net, save_dir, epoch_info, gpu_id):
    save_filename = '%s.pth' %epoch_info
    save_path = os.path.join(save_dir, save_filename)
    if len(gpu_id) > 1:
        checkpoint = {'state_dict': net.module.state_dict()}
    else:
        checkpoint = {'state_dict': net.state_dict()}
    torch.save(checkpoint, save_path)


def update_learning_rate(cur_lr, decay_rate, optimizer):
    lr = cur_lr * decay_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Update learning rate: %f -> %f' %(cur_lr, lr))
    return lr