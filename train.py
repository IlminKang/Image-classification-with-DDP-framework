import os
import sys
import argparse
import wandb
import time
import asyncio

import torch
import torch.nn as nn

from utils import general, losses, multiprocess
from dataset import loader
from models import get_model
from core import engine



def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Project Name", add_help=add_help)


    # Training parameters
    parser.add_argument('--gpu', type=str, default='0,1,2,3', help='usable device number')
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument('--seed', type=int, default=None, help='seed')
    parser.add_argument('--start_epoch', type=int, default=0, help='starting point of epochs')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--loss', type=str, default='cross_entropy', help='objective function')
    parser.add_argument("--opt", type=str, default='Adam', help="optimizer")
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M", help="momentum")
    parser.add_argument("--wd", "--weight_decay", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay")
    parser.add_argument("--lr_scheduler", default="steplr", type=str, help="lr scheduler (default: steplr)")
    parser.add_argument("--lr_step", default=100, type=int,help="decrease lr every step-size epochs (steplr scheduler only)")
    parser.add_argument("--lr_steps", default=[10, 30, 50], nargs="+", type=int,help="decrease lr every step-size epochs (multisteplr scheduler only)")
    parser.add_argument("--lr_gamma", default=0.1, type=float,help="decrease lr by a factor of lr-gamma")
    parser.add_argument('--save_dir', type=str, default='./results', help='output save dir')
    parser.add_argument('--save', action='store_true', help='save models')
    parser.add_argument("--resume", default=False, type=bool, help="restart training")
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument("--wandb", action='store_true', help="use wandb")


    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # clustering parameters
    parser.add_argument('--output_k', type=int, default=20, help='number of clusters')
    parser.add_argument('--cluster_mode', type=str, default='')
    parser.add_argument('--patch_size', type=int, default=16, help='size of the patches')

    # Encoder parameters
    parser.add_argument('--encoder', type=str, default='resnet50', help='base encoder type')
    parser.add_argument('--encoder_weight', type=str, default='', help='path to encoder weight')

    # Distributed training parameters
    parser.add_argument("--world-size", default=4, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", action="store_true")

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='cifar10', help='type of dataset [cifar10, cifar100, miniImageNet, tinyImageNet, places365]')
    parser.add_argument('--data_dir', type=str, default=None, help='path to dataset directory')
    parser.add_argument('--class_num', type=int, default=None, help='number of class labels')
    parser.add_argument("-b", "--batch_size", default=16, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument('--num_workers', type=int, default=4, help='size of workers')
    parser.add_argument('--norm', action='store_true', help='normalize image')

    return parser

def main(args):

    # settings
    device = torch.device(args.device)

    multiprocess.init_distributed_mode(args)
    args.local_rank = int(os.environ["LOCAL_RANK"])

    print('>>>Using Distributed Data Parallel...')

    if args.seed:
        current_seed = general.fix_seed(args.seed)
    else:
        current_seed = 'RandomSeed'

    general.dataset_qualification(args)
    general.model_qualification(args)

    if args.local_rank == 0:
        model_name = general.get_model_name(args, current_seed)


    # weights and bias
    if args.wandb and args.local_rank == 0:
        project = f"Project Name"
        wandb.init(project=project, entity="ilmin", name=model_name)
    else:
        pass

    print('Done!\n')



    # load datasets
    print(f'>>>Using {args.dataset}, loading data from "{args.data_dir}"...')
    print(f'\t|-Image size: {args.img_size} / Patch size: {args.patch_size}')

    train_set = loader.get_dataset(args, split='train')
    val_set = loader.get_dataset(args, split='val')
    test_set = loader.get_dataset(args, split='test')

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, shuffle=False)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              sampler=test_sampler)
    if val_set:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
        val_loader = torch.utils.data.DataLoader(val_set,
                                                 batch_size=args.batch_size,
                                                 num_workers=args.num_workers,
                                                 pin_memory=True,
                                                 sampler=val_sampler)
    print('Done!\n')



    # loading network
    print(">>>Creating network...")
    model = get_model.load_model(args, device)
    model.to(device)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])  # , find_unused_parameters=True
    model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = general.get_optimizer(args, params)
    lr_scheduler = general.get_scheduler(args, optimizer)
    criterion = losses.get_loss(args)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    if args.resume:
        print(f'continue training from {args.resume}...')
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    print('Done!\n')


    print(">>>Start training...")
    start_time = time.time()
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)

            train_loss, train_top1, train_top5 = engine.train_epoch(model, optimizer, criterion, train_loader, device, epoch,
                                                              print_freq=200, scaler=scaler)
            test_loss, test_top1, test_top5 = engine.eval_epoch(model, criterion, test_loader, device, epoch,
                                                                     print_freq=20)
            lr_scheduler.step()

            if args.wandb and args.local_rank == 0:
                wandb.log({'train_loss': train_loss,
                           'train_top1': train_top1,
                           'train_top5': train_top5,
                           'test_loss': test_loss,
                           'test_top1': test_top1,
                           'test_top5': test_top5,
                           })

            if test_top1 > best_acc:
                best_acc = test_top1
                if args.local_rank == 0 and args.save:
                    general.save_checkpoint(state={'epoch': epoch + 1,
                                                   'args': args,
                                                   'model': model_without_ddp.state_dict(),
                                                   'optimizer': optimizer.state_dict(),
                                                   'lr_scheduler': lr_scheduler.state_dict(),
                                                   'scaler': scaler.state_dict()}, is_best=True, save_root=args.save_dir, filename='model_best.pth')

    if args.wandb and args.local_rank == 0:
        wandb.log({'best_acc': best_acc,
                   })

    if args.local_rank == 0 and args.save:
        general.save_checkpoint(state={'epoch': epoch + 1,
                                       'args': args,
                                       'model': model_without_ddp.state_dict(),
                                       'optimizer': optimizer.state_dict(),
                                       'lr_scheduler': lr_scheduler.state_dict(),
                                       'scaler': scaler.state_dict()}, is_best=True, save_root=args.save_dir, filename='model_last.pth')


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)


