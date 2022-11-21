import torch
import torch.nn as nn
import sys
import os


def get_loss(args):
    if args.loss == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise TypeError(f'Invalid loss! {args.loss}')

    return criterion
