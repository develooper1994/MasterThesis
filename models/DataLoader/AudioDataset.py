# original https://github.com/ericlearning/generative-unconditional/blob/master/dataset.py
import os
import random

import librosa
import numpy as np
import pescador
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio import datasets, transforms

from config import DATASET_NAME, OUTPUT_PATH, WINDOW_LENGHT, FS, EPOCHS
from models.utils.BasicUtils import make_path
# from models.utils.WaveGAN_utils import LOGGER as L  # ImportError: cannot import name 'LOGGER' from
# 'models.utils.WaveGAN_utils'
from models.utils import WaveGAN_utils


class AudioDataset(Dataset):
    def __init__(self, input_dir=None, output_dir=None, input_transform=None):
        self.input_dir = input_dir
        if input_dir is None:
            self.input_dir = DATASET_NAME
        self.output_dir = output_dir
        if output_dir is None:
            self.output_dir = OUTPUT_PATH
        self.input_transform = input_transform

        self.audio_name_list = self.get_all_audio_filepaths

    def __len__(self):
        return len(self.audio_name_list)

    def __getitem__(self, idx):
        # TODO Test and use.
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # input_audio = np.load(os.path.join(self.input_dir, self.audio_name_list[idx]))
        # point = random.randint(0, input_audio.shape[0] - self.num_samples)
        # input_audio = input_audio[point:point + self.num_samples] / 32768.0
        # input_audio = torch.from_numpy(input_audio)
        # input_audio = input_audio.view(1, -1).float()
        audio_name = self.audio_name_list[idx]
        input_audio = torchaudio.load(audio_name)

        # point = torch.randint(0, input_audio.shape[0] - self.num_samples, (1,))
        point = random.randint(0, input_audio.shape[0] - self.num_samples)
        input_audio = input_audio[point:point + self.num_samples] / 32768.0
        # input_audio = torch.from_numpy(input_audio)
        input_audio = input_audio.view(1, -1).float()

        if self.input_transform is not None:
            input_audio = self.input_transform(input_audio)

        return input_audio, 0

    @property
    def get_all_audio_filepaths(self):
        """
        Returns all available audio file paths
        :param input_dir: audio dataset directory
        :return: all available audio file paths
        """
        return [os.path.join(root, fname)
                for (root, dir_names, file_names) in os.walk(self.input_dir, followlinks=True)
                for fname in file_names
                if fname.lower().endswith('.wav') or fname.lower().endswith('.mp3')  # you can add file types with
                # "or fname.lower().endswith('.mp3')"
                ]

    # TODO: replace librosa with torchaudio
    def save_samples(self, epoch, epoch_samples, fs=FS) -> None:
        """
        Save output samples for each iteration to examine progress
        :param epoch: iteration number
        :param fs: sampling frequency
        :return: None
        """
        sample_dir = make_path(os.path.join(self.output_dir, str(epoch)))

        for idx, sample in enumerate(epoch_samples):
            output_path = os.path.join(sample_dir, "{}.wav".format(idx + 1))
            sample = sample[0]
            librosa.output.write_wav(output_path, sample, fs)

    # Adapted from @jtcramer https://github.com/jtcramer/wavegan/blob/master/sample.py.
    @staticmethod
    def sample_generator(filepath, window_length=WINDOW_LENGHT, fs=FS):
        """
        Audio sample generator from dataset
        :param filepath: Full path for dataset
        :param window_length: windowing lenght.
         function sampling with a windowing technique.
        :param fs: sampling frequency
        :return: gives a generator to iterate over big dataset
            :type: return type is generator
        """
        try:
            audio_data, _ = librosa.load(filepath, sr=fs)

            # Clip magnitude
            max_mag = np.max(np.abs(audio_data))
            if max_mag > 1:
                audio_data /= max_mag
        except Exception as e:
            WaveGAN_utils.LOGGER.error("Could not load {}: {}".format(filepath, str(e)))
            raise StopIteration

        # Pad audio to >= window_length.
        audio_len = len(audio_data)
        if audio_len < window_length:
            pad_length = window_length - audio_len
            left_pad = pad_length // 2
            right_pad = pad_length - left_pad

            audio_data = np.pad(audio_data, (left_pad, right_pad), mode='constant')
            audio_len = len(audio_data)

        while True:
            if audio_len == window_length:
                # If we only have a single 1*window_length audio, just yield.
                sample = audio_data
            else:
                # Sample a random window from the audio
                start_idx = np.random.randint(0, (audio_len - window_length) // 2)
                end_idx = start_idx + window_length
                sample = audio_data[start_idx:end_idx]

            sample = sample.astype('float32')
            assert not np.any(np.isnan(sample))

            yield {'X': sample}

    def batch_generator(self, audio_path_list, batch_size):
        """
        Generates batches to input algorithm(NN)
            batch <-> bunch of samples inputs algorithm(NN) one at a time
        :param audio_path_list: list of all paths of audio dataset
        :param batch_size: size(lenght) of batch
        :return: generated batch
        """
        streamers = []
        for audio_path in audio_path_list:
            s = pescador.Streamer(self.sample_generator, audio_path)
            streamers.append(s)

        mux = pescador.ShuffledMux(streamers)
        batch_gen = pescador.buffer_stream(mux, batch_size)

        return batch_gen

    # TODO: replace with torchaudio
    def split_data(self, audio_path_list, valid_ratio, test_ratio, batch_size):
        """
        Split data into *Train, *Validation(dev), *Test
        :param audio_path_list: list of all paths of audio dataset
        :param valid_ratio: *Validation dataset split radio.
            *Validation data has to came from same distribution with *Train data
        :param test_ratio: *Test dataset split radio.
            Test data can be very big so that test with little test samples
        :param batch_size: size(lenght) of batch
        :return: tuple of splited into (*Train, *Validation, *Test)
        """
        num_files = torch.tensor(len(audio_path_list), dtype=torch.int)
        num_valid = int(torch.ceil(num_files * valid_ratio))
        num_test = int(torch.ceil(num_files * test_ratio))
        num_train = num_files - num_valid - num_test

        if num_valid <= 0 or num_test <= 0 or num_train <= 0:
            WaveGAN_utils.LOGGER.error("Please download DATASET '{}' and put it under current path !".format(DATASET_NAME))

        # Random shuffle the audio_path_list for splitting.
        random.shuffle(audio_path_list)

        valid_files = audio_path_list[:num_valid]
        test_files = audio_path_list[num_valid:num_valid + num_test]
        train_files = audio_path_list[num_valid + num_test:]
        train_size = len(train_files)

        train_data = self.batch_generator(train_files, batch_size)
        valid_data = self.batch_generator(valid_files, batch_size)
        test_data = self.batch_generator(test_files, batch_size)

        return train_data, valid_data, test_data, train_size

    def split_manage_data(self, arguments, batch_size):
        train_data, valid_data, test_data, train_size = self.split_data(self.audio_name_list,
                                                                        arguments['valid-ratio'],
                                                                        arguments['test-ratio'],
                                                                        batch_size)
        TOTAL_TRAIN_SAMPLES = train_size
        BATCH_NUM = TOTAL_TRAIN_SAMPLES // batch_size

        train_iter, valid_iter, test_iter = iter(train_data), iter(valid_data), iter(test_data)

        return BATCH_NUM, train_iter, valid_iter, test_iter


class Sc09(AudioDataset):
    def __init__(self):
        super(Sc09, self).__init__()


class Piano:
    def __init__(self):
        super(Piano, self).__init__()
