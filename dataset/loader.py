import os
import torch
import json
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import InterpolationMode

def get_dataset(args, split='train', eval=True):
    '''
    :param args: hyper-parameters
    :param split: split
    :return: dataset instance
    '''
    if split not in ['train', 'val', 'test']:
        raise TypeError(f'Invalid split type! {split}')

    if args.dataset.lower() == 'cifar10':
        from dataset.cifar10 import Cifar10Dataset as Dataset_

        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        if split == 'val':
            return None

    elif args.dataset.lower() == 'cifar100':
        from dataset.cifar100 import Cifar100Dataset as Dataset_

        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        if split == 'val':
            return None

    elif args.dataset.lower() == 'tinyimagenet':
        from dataset.tiny_imagenet import ImageNetDataset as Dataset_

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    elif args.dataset.lower() == 'miniimagenet':
        pass

    elif args.dataset.lower() == 'imagenet':
        from dataset.imagenet import ImageNetDataset as Dataset_

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    elif args.dataset.lower() == 'places365':
        from dataset.places365 import Places365Dataset as Dataset_

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if split == 'val':
            return None

        if split == 'test':
            split = 'val'

        raise TypeError(f'Invalid dataset! : {args.dataset}')


    root_dir = args.data_dir
    patch_size = args.patch_size

    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.Normalize(mean, std),
                                    ])

    dataset = Dataset_(root_dir=root_dir,
                           transform=transform,
                           patch_size=patch_size,
                           split=split)

    print(f'\t|-{args.dataset} {split}: {len(dataset)}')
    return dataset
