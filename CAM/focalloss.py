###################################################################
##### This is focal loss class for multi-label classification #####
###################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F


# First implementation, referred https://github.com/andrijdavid/FocalLoss/blob/master/focalloss.py
class FocalLoss(nn.modules.loss._WeightedLoss):

    def __init__(self, gamma=2, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', pos_weight=None):
        super(FocalLoss, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.pos_weight = pos_weight

    def forward(self, input, target):
        # inputs and targets are assumed to be BatchxClasses
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)

        # compute the negative likelihood
        logpt = - F.binary_cross_entropy_with_logits(input, target, pos_weight=self.pos_weight, reduction=self.reduction)
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -((1 - pt)**self.gamma) * logpt
        return focal_loss


## Another implementation, referred to https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
class FocalLoss2(nn.Module):
    def __init__(self, pos_weight=None, gamma=1, reduction='mean'):
        super(FocalLoss2, self).__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        '''
        params: inputs - torch.Tensor (B,K)
        params: targets - torch.Tensor (B,K)
        '''
        assert len(inputs.shape) == len(targets.shape)
        assert inputs.size(0) == targets.size(0)
        assert inputs.size(1) == targets.size(1)

        log_pt = - F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=self.pos_weight) #(B,K)
        pt = torch.exp(log_pt)

        focal_loss = - ((1 - pt)**self.gamma) * log_pt

        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            raise Exception("The reduction option can only be 'none' | 'mean' | 'sum'!")