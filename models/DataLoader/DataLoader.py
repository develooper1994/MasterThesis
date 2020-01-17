from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.DataLoader.AudioDataset import AudioDataset, Sc09, Piano
from config import OUTPUT_PATH


# original: https://github.com/ericlearning/generative-unconditional/blob/master/dataset.py
class Dataset:
    def __init__(self, train_dir, basic_types: str = None, shuffle=True):
        self.train_dir = train_dir
        self.basic_types = basic_types
        self.shuffle = shuffle

    def get_loader(self, sz, bs, get_size=False, data_transform=None, num_workers=1, audio_sample_num=None):
        returns = tuple()
        if self.basic_types is None:
            if data_transform is None:
                data_transform = transforms.Compose([
                    transforms.Resize(sz),
                    transforms.CenterCrop(sz),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])

            train_dataset = datasets.ImageFolder(self.train_dir, data_transform)
            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=self.shuffle, num_workers=num_workers)

            train_dataset_size = len(train_dataset)
            size = train_dataset_size

            returns = train_loader
            if get_size:
                returns = returns + (size,)

        elif self.basic_types.lower() == 'mnist':
            data_transform = transforms.Compose([
                transforms.Resize(sz),
                transforms.CenterCrop(sz),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

            train_dataset = datasets.MNIST(self.train_dir, train=True, download=True, transform=data_transform)
            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=self.shuffle, num_workers=num_workers)

            train_dataset_size = len(train_dataset)
            size = train_dataset_size

            returns = train_loader
            if get_size:
                returns = returns + (size,)

        elif self.basic_types.lower() == 'cifar10':
            data_transform = transforms.Compose([
                transforms.Resize(sz),
                transforms.CenterCrop(sz),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

            train_dataset = datasets.CIFAR10(self.train_dir, train=True, download=True, transform=data_transform)
            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=self.shuffle, num_workers=num_workers)

            train_dataset_size = len(train_dataset)
            size = train_dataset_size

            returns = tuple(train_loader)
            if get_size:
                returns = returns + (size,)

        ## audio datasets
        elif self.basic_types.lower() == 'sc09':
            print("Not completed yet!!!")
            train_dataset = AudioDataset(self.train_dir, data_transform, audio_sample_num)
            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=self.shuffle, num_workers=num_workers)

            returns = train_loader

        elif self.basic_types.lower() == 'piano':
            print("Not completed yet!!!")
            train_dataset = AudioDataset(self.train_dir, data_transform, audio_sample_num)
            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=self.shuffle, num_workers=num_workers)

            returns = train_loader

        elif self.basic_types.lower() == 'audio':
            print("Not completed yet!!!")
            train_dataset = AudioDataset(self.train_dir, OUTPUT_PATH, data_transform, audio_sample_num)
            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=self.shuffle, num_workers=num_workers)

            returns = train_loader

        return returns


