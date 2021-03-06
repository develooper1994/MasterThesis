# original https://github.com/ericlearning/generative-unconditional/blob/master/dataset.py
import os
import random

import librosa
import numpy as np
import pescador
import torch
import torchaudio
from torch.utils.data import Dataset

from not_functional.config import target_signals_dir, window_length, sampling_rate
from not_functional.models.utils.BasicUtils import make_path
# from models.utils.wave_gan_utils import LOGGER as L  # ImportError: cannot import name 'LOGGER' from
# 'models.utils.wave_gan_utils'
from not_functional.models.utils.WaveGANUtils import WaveGANUtils


class AudioDataset(Dataset):
    def __init__(self, input_dir=None, output_dir=None, input_transform=None, audio_number_samples=sampling_rate):
        self.input_dir = input_dir
        if input_dir is None:
            self.input_dir = target_signals_dir
        self.output_dir = output_dir
        if output_dir is None:
            self.output_dir = output_dir
        self.input_transform = input_transform
        self.num_samples = audio_number_samples

        self.audio_name_list = self.get_all_audio_filepaths
        self.audio_label_list = self.get_all_audio_labels

    # @torch.no_grad()
    def __len__(self):
        return len(self.audio_name_list)

    # @torch.no_grad()
    def __getitem__(self, idx):
        # TODO Test and use.
        print("Not done yet!!!")
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # input_audio = np.load(os.path.join(self.input_dir, self.audio_name_list[idx]))
        # point = random.randint(0, input_audio.shape[0] - self.num_samples)
        # input_audio = input_audio[point:point + self.num_samples] / 32768.0
        # input_audio = torch.from_numpy(input_audio)
        # input_audio = input_audio.view(1, -1).float()
        audio_name = self.audio_name_list[idx]
        input_audio, sample_rate = torchaudio.load(audio_name)

        point = torch.randint(0, int(input_audio.shape[1] - self.num_samples), (1,))
        # point = random.randint(0, input_audio.shape[1] - self.num_samples)
        input_audio = input_audio[0, point:point + self.num_samples] / 32768.0
        # input_audio = torch.from_numpy(input_audio)
        input_audio = input_audio.view(1, -1).float() / 32768.0

        if self.input_transform is not None:
            input_audio = self.input_transform(input_audio)

        return input_audio, self.get_all_audio_labels[idx]  # self.audio_label_list[idx]

    @property
    def get_all_audio_filepaths(self):
        """
        Returns all available audio file paths
        :param input_dir: audio dataset directory
        :return: all available audio file paths
        """
        file_types = ['.wav', '.mp3']  # you can add file types with
        return [os.path.join(root, fname)
                for (root, dir_names, file_names) in os.walk(self.input_dir, followlinks=True)
                for fname in file_names
                for file_type in file_types
                if fname.lower().endswith(file_type)
                ]

    @property
    def get_all_audio_labels(self):
        # L[0].split('/')[-1].split('_')[0]  # split pattern to get label
        # one_data.split('/')[-1].split('_')[0]
        paths = self.audio_name_list
        return [label.split('/')[-1].split("_")[0]
                for label in paths]

    # TODO: replace librosa with torchaudio
    def save_samples(self, epoch, epoch_samples, fs=sampling_rate) -> None:
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
    def sample_generator(filepath, label, window_length=window_length, fs=sampling_rate):
        """
        Audio sample generator from dataset
        :param filepath: Full path for dataset
        :param label: label of data
        :param window_length: windowing lenght.
         function sampling with a windowing technique.
        :param fs: sampling frequency
        :return: gives a generator to iterate over big dataset
            :type: return type is generator
        """
        try:
            audio_data, sampling_rate = librosa.load(filepath, sr=fs)
            # label = L[0].split('/')[-1].split('_')[0]

            # Clip magnitude
            # TODO: tey to convery numpy to pytorch
            max_mag = np.max(np.abs(audio_data))
            if max_mag > 1:
                audio_data /= max_mag
        except Exception as e:
            WaveGANUtils.LOGGER.error("Could not load {}: {}".format(filepath, str(e)))
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

            yield {'X': sample, 'Y': label}  # TODO: insert label info into dict.

    def batch_generator(self, audio_path_list, audio_label_list, batch_size):
        """
        Generates batches to input algorithm(NN)
            batch <-> bunch of samples inputs algorithm(NN) one at a time
        :param audio_path_list: list of all paths of audio dataset
        :param batch_size: size(lenght) of batch
        :return: generated batch
        """
        # TODO: Make it closure form to get some speed
        streamers = []
        for (audio_path, audio_label) in zip(audio_path_list, audio_label_list):
            s = pescador.Streamer(self.sample_generator, audio_path, audio_label)
            streamers.append(s)

        mux = pescador.ShuffledMux(streamers)
        return pescador.buffer_stream(mux, batch_size)

    # TODO: replace with torchaudio
    def split_data(self, audio_path_list, audio_label_list, valid_ratio, test_ratio, batch_size):
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
            WaveGANUtils.LOGGER.error(
                "Please download DATASET '{}' and put it under current path !".format(target_signals_dir))

        # Random shuffle the audio_path_list for splitting.
        random.shuffle(audio_path_list)

        train_files = audio_path_list[num_valid + num_test:]
        valid_files = audio_path_list[:num_valid]
        test_files = audio_path_list[num_valid:num_valid + num_test]

        train_labels = audio_label_list[num_valid + num_test:]
        valid_labels = audio_label_list[:num_valid]
        test_labels = audio_label_list[num_valid:num_valid + num_test]
        train_size = len(train_files)

        train_data = self.batch_generator(train_files, train_labels, batch_size)
        valid_data = self.batch_generator(valid_files, valid_labels, batch_size)
        test_data = self.batch_generator(test_files, test_labels, batch_size)

        return train_data, valid_data, test_data, train_size

    # @torch.no_grad()
    def split_manage_data(self, arguments, batch_size):
        train_data, valid_data, test_data, train_size = self.split_data(self.audio_name_list,
                                                                        self.audio_label_list,
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
