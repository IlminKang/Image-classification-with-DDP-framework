import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets


class Places365Dataset(Dataset):
    def __init__(self, root_dir, image_size, transform, transform_eval, transform_ori, transform_aug, patch_size,
                 split='train'):
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transform
        self.transform_eval = transform_eval
        self.transform_ori = transform_ori
        self.transform_aug = transform_aug
        self.patch_size = patch_size

        self.root_dir = f'{root_dir}/{split}'
        self.data = datasets.ImageFolder(self.root_dir, transforms.Compose([transforms.ToTensor(), ]))

    def __len__(self):
        return len(self.data)

    def class_to_idx(self):
        return self.data.class_to_idx

    def classes(self):
        return self.data.classes

    def __getitem__(self, idx):
        image, cls = self.data[idx]

        image = self.transform(image)

        return image, cls
