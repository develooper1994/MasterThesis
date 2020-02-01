# Subset of the Speech Commands dataset: https://research.googleblog.com/2017/08/launching-speech-commands-dataset.html
# License: https://creativecommons.org/licenses/by/4.0/
import os
import random

import librosa
import numpy as np
# from torch.autograd import Variable
# pescador has no cuda and TorchJIT support
import pescador
import torch
import torchaudio as to

from config import target_signals_dir, window_length, sampling_rate
from models.utils.BasicUtils import make_path
from models.utils.WaveGANUtils import WaveGANUtils  # LOGGER


# ============================================================
# TODO: replace librosa with torchaudio
def save_samples(epoch_samples, epoch, output_dir, fs=sampling_rate) -> None:
    """
    Save output samples for each iteration to examine progress
    :param epoch_samples: samples for each iteration
    :param epoch: iteration number
    :param output_dir: output directory
    :param fs: sampling frequency
    :return: None
    """
    sample_dir = make_path(os.path.join(output_dir, str(epoch)))

    for idx, sample in enumerate(epoch_samples):
        output_path = os.path.join(sample_dir, "{}.wav".format(idx + 1))
        sample = sample[0]
        librosa.output.write_wav(output_path, sample, fs)


# TODO: replace librosa with torchaudio
# Adapted from @jtcramer https://github.com/jtcramer/wavegan/blob/master/sample.py.
def sample_generator(filepath, window_length=window_length, fs=sampling_rate):
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

        yield {'X': sample}


# TODO: replace with torchaudio
def get_all_audio_filepaths(audio_dir):
    """
    Returns all available audio file paths
    :param audio_dir: audio dataset directory
    :return: all available audio file paths
    """
    return [os.path.join(root, fname)
            for (root, dir_names, file_names) in os.walk(audio_dir, followlinks=True)
            for fname in file_names
            if fname.lower().endswith('.wav') or fname.lower().endswith('.mp3')
            ]


# TODO: replace with torchaudio
def batch_generator(audio_path_list, batch_size):
    """
    Generates batches to input algorithm(NN)
        batch <-> bunch of samples inputs algorithm(NN) one at a time
    :param audio_path_list: list of all paths of audio dataset
    :param batch_size: size(lenght) of batch
    :return: generated batch
    """
    streamers = []
    for audio_path in audio_path_list:
        s = pescador.Streamer(sample_generator, audio_path)
        streamers.append(s)

    mux = pescador.ShuffledMux(streamers)
    batch_gen = pescador.buffer_stream(mux, batch_size)

    return batch_gen


# TODO: replace with torchaudio
def split_data(audio_path_list, valid_ratio, test_ratio, batch_size):
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
        WaveGANUtils.LOGGER.error("Please download DATASET '{}' and put it under current path !".format(target_signals_dir))

    # Random shuffle the audio_path_list for splitting.
    random.shuffle(audio_path_list)

    valid_files = audio_path_list[:num_valid]
    test_files = audio_path_list[num_valid:num_valid + num_test]
    train_files = audio_path_list[num_valid + num_test:]
    train_size = len(train_files)

    train_data = batch_generator(train_files, batch_size)
    valid_data = batch_generator(valid_files, batch_size)
    test_data = batch_generator(test_files, batch_size)

    return train_data, valid_data, test_data, train_size


def split_manage_data(audio_dir, arguments, batch_size):
    audio_paths = get_all_audio_filepaths(audio_dir)
    train_data, valid_data, test_data, train_size = split_data(audio_paths,
                                                               arguments['valid-ratio'],
                                                               arguments['test-ratio'],
                                                               batch_size)
    TOTAL_TRAIN_SAMPLES = train_size
    BATCH_NUM = TOTAL_TRAIN_SAMPLES // batch_size

    train_iter, valid_iter, test_iter = iter(train_data), iter(valid_data), iter(test_data)

    return BATCH_NUM, train_iter, valid_iter, test_iter
