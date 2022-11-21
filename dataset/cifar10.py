import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


class Cifar10Dataset(Dataset):
    def __init__(self, root_dir, transform, patch_size, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.patch_size = patch_size

        if split == 'train':
            train = True
        elif split == 'val':
            raise TypeError(f'Invalid split type for CIFAR10! : {split}')
        else:
            train = False

        self.data = torchvision.datasets.CIFAR10(root=self.root_dir, train=train, download=True,
                                                 transform=transforms.Compose([transforms.Resize((self.image_size, self.image_size)),
                                                                               transforms.ToTensor(),
                                                                               ]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, cls = self.data[idx]

        image = self.transform(image)

        return image, cls