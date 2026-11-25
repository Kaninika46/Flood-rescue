# Copied from https://github.com/SpaceNetChallenge/SpaceNet8 (Apache-2.0),
# commit 30ca30c2530c6a234d040f60a99a7366ff62d756, file baseline/core/losses.py.
# Changes: none.
# Copyright of the original: the SpaceNet 8 authors; licence: third_party/spacenet8/LICENSE.
# ruff: noqa
""" loss functions. also found in cresi repo """
import torch

def soft_dice_loss(outputs, targets, per_image=False):
    '''
    From cannab sn4
    '''
    
    batch_size = outputs.size()
    # batch_size = outputs.size()[0]
    eps = 1e-5
    if not per_image:
        batch_size = 1
    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps
    loss = (1 - (2 * intersection + eps) / union).mean()
    return loss


def focal(outputs, targets, gamma=2,  ignore_index=255):
    '''From cannab sn4'''
    outputs = outputs.contiguous()
    targets = targets.contiguous()
    eps = 1e-8
    non_ignored = targets.view(-1) != ignore_index
    targets = targets.view(-1)[non_ignored].float()
    outputs = outputs.contiguous().view(-1)[non_ignored]
    outputs = torch.clamp(outputs, eps, 1. - eps)
    targets = torch.clamp(targets, eps, 1. - eps)
    pt = (1 - targets) * (1 - outputs) + targets * outputs
    return (-(1. - pt) ** gamma * torch.log(pt)).mean()
