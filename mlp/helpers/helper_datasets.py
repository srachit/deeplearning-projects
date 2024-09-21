import torch
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import datasets, transforms


def get_dataloaders_mnist(batch_size, num_workers=0,
                          validation_fraction=None,
                          train_transforms=None,
                          test_transforms=None):
    if not train_transforms:
        train_transforms = transforms.ToTensor()

    if not test_transforms:
        test_transforms = transforms.ToTensor()

    train_dataset = datasets.MNIST(root="data",
                                   train=True,
                                   transform=train_transforms,
                                   download=True)

    valid_dataset = datasets.MNIST(root="data",
                                   train=True,
                                   transform=test_transforms)

    test_dataset = datasets.MNIST(root="data",
                                  train=False,
                                  transform=test_transforms)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)

    if validation_fraction:
        train_dataset_size = len(train_dataset)
        num = int(validation_fraction * train_dataset_size)
        data_range = train_dataset_size - num

        train_indices = torch.arange(0, data_range)
        valid_indices = torch.arange(data_range, train_dataset_size)

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  sampler=valid_sampler)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  sampler=train_sampler,
                                  drop_last=True)

        return train_loader, valid_loader, test_loader
    else:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  shuffle=True)
        return train_loader, test_loader
