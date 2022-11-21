import os
import wandb
import torch
import random
import numpy as np
import shutil


def dataset_qualification(args):
    '''
    :param args: dataset
    :return: number of class, location of dataset, image size
    '''
    if args.dataset.lower() == 'cifar10':
        args.class_num = 10
        args.data_dir = '/home/t1_u1/ilmin/workspace/datas/cifar-10'
        args.img_size = 32
    elif args.dataset.lower() == 'cifar100':
        args.class_num = 100
        args.data_dir = '/home/t1_u1/ilmin/workspace/datas/cifar-100'
        args.img_size = 32
    elif args.dataset.lower() == 'tinyimagenet':
        args.class_num = 200
        args.data_dir = '/home/t1_u1/ilmin/workspace/datas/tiny-imagenet-200'
        args.img_size = 64
    elif args.dataset.lower() == 'miniimagenet':
        args.class_num = 100
        args.data_dir = '/home/t1_u1/ilmin/workspace/datas/miniImageNet'
        args.img_size = 256
    elif args.dataset.lower() == 'imagenet':
        args.class_num = 1000
        args.data_dir = '/data/ilmin/datas/ILSVRC2012/imagenet/'
        args.img_size = 256
    elif args.dataset.lower() == 'places365':
        args.class_num = 365
        args.data_dir = '/data/ilmin/datas/places365_standard_small'
        args.img_size = 256
    else:
        raise TypeError(f'Invalid dataset! : {args.dataset}')


def model_qualification(args):
    '''
    :param args: model hyper-parameters
    :return:
    '''


def get_optimizer(args, params):
    '''
    :param args: optimizer hyper-parameters
    :param params: model parameters
    :return: optimizer
    '''
    print(f'\t|-Optimizer: {args.opt}')
    if args.opt.lower() == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        print(f'\t\t|-lr:{args.lr}, momentum:{args.momentum}, weight_decay:{args.weight_decay}')
    elif args.opt.lower() == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        print(f'\t\t|-lr:{args.lr}, weight_decay:{args.weight_decay}')
    else:
        raise TypeError(f'Invalid optimizer type! : {args.opt.lower()}')

    return optimizer

def get_scheduler(args, optimizer):
    '''
    :param args: scheduler hyper-parameter
    :return: scheduler
    '''
    print(f'\t|-Scheduler: {args.lr_scheduler}')
    if args.lr_scheduler.lower() == "steplr":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
        print(f'\t\t|-lr_step:{args.lr_step}, lr_gamma:{args.lr_gamma}')
    elif args.lr_scheduler.lower() == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
        print(f'\t\t|-lr_steps:{args.lr_steps}, lr_gamma:{args.lr_gamma}')
    else:
        raise TypeError(f'Invalid Scheduler type! : {args.lr_scheduler.lower()}')

    return lr_scheduler

def fix_seed(seed):
    '''
    :param seed: seed
    :return: current seed string type
    '''
    print(f'>>>Fixing seed : {seed}')
    random_seed = seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    return str(random_seed)


def get_model_name(args, current_seed):
    '''
    :param args: hyper-parameters
    :param current_seed: current seed state
    :return: model name
    '''
    names = []
    model_name = '_'.join(names)
    print(f'>>>Training {model_name}...')

    make_save_dir(args, model_name)

    return model_name


def make_save_dir(args, model_name):
    '''
    :param args: hyper-parameters
    :param model_name: model name
    :return:
    '''
    dir_path = f'{args.save_dir}/{model_name}'
    if not os.path.isdir(f'{args.save_dir}'):
        os.mkdir(f'{args.save_dir}')

    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    print(f'\t|-Saving model to {dir_path}')
    args.save_dir = dir_path


def save_checkpoint(state, is_best, save_root, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        # shutil.copyfile(filename, f'{save_root}/{filename}')
        print(f'Saving checkpoint to {save_root}/{filename}...')
        torch.save(state, f'{save_root}/{filename}')
