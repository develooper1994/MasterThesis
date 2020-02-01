import os
import time
from typing import Iterator, Dict, Any, Union, List, NoReturn, Generator

import numpy as np
import torch
from numpy import ndarray
from numpy.core._multiarray_umath import ndarray
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# my modules
from config import output_dir, device, target_signals_dir
from models.DataLoader.AudioDataset import AudioDataset
from models.utils.BasicUtils import get_recursive_files, create_stream_reader


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
            train_dataset = AudioDataset(self.train_dir, output_dir, data_transform, audio_sample_num)
            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=self.shuffle, num_workers=num_workers)

            returns = train_loader

        return returns


#############################
# Creating Data Loader and Sampler
#############################
class WavDataLoader:
    data_iter: Iterator[Dict[Any, Union[Union[ndarray, ndarray, ndarray], Any]]]
    signal_paths: List[Union[bytes, str]]

    def __init__(self, folder_path: object, audio_extension: object = 'wav') -> NoReturn:
        self.signal_paths, self.signal_label = get_recursive_files(folder_path, audio_extension)
        self.data_iter = None
        self.initialize_iterator()

    def initialize_iterator(self) -> NoReturn:
        data_iter: Generator[Dict[Any, Union[ndarray, Any]], Any, Any] = create_stream_reader(self.signal_paths, self.signal_label)
        self.data_iter = iter(data_iter)

    def __len__(self) -> int:
        return len(self.signal_paths)

    def numpy_to_tensor(self, numpy_array, label):
        numpy_array = numpy_array[:, np.newaxis, :]
        return torch.Tensor(numpy_array).to(device), label

    def __iter__(self):
        return self

    def __next__(self):
        it: Dict[Any, Union[ndarray, Any]] = next(self.data_iter)
        return self.numpy_to_tensor(it['single'], it['label'])


if __name__ == '__main__':
    # import time

    start: float = time.time()
    print(time.time() - start)
    train_loader: WavDataLoader = WavDataLoader(os.path.join(target_signals_dir, 'train'))
    start: float = time.time()
    for i in range(7):
        x = next(train_loader)
    print(time.time() - start)