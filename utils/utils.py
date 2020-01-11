import json
import os
import time
import datetime
import math
import torch
import random
import logging
import librosa
import argparse
# pescador has no cuda and TorchJIT support
import pescador  # https://github.com/pescadores/pescador
import numpy as np

from models.wavegan import WaveGANGenerator, WaveGANDiscriminator
from utils.WaveGAN_utils import *
from utils.config import EPOCHS, BATCH_SIZE
from utils.config import SAMPLE_EVERY, SAMPLE_NUM
from utils.config import DATASET_NAME, OUTPUT_PATH
from torch import autograd, optim
# from torch.autograd import Variable
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

# File Logger Configfuration
LOGGER = logging.getLogger('wavegan')
LOGGER.setLevel(logging.DEBUG)


def make_path(output_path):
    """
    Create folder
    :param output_path: full paths to create folders
    :return: created folders with full paths.
    """
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    return output_path


# Dataset folder path creatation configuration
traindata = DATASET_NAME
output = make_path(OUTPUT_PATH)


def time_since(since):
    """
    Measures time in human readable format.
    :param since: point in time to measure
    :return: difference between two time points
    """
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# ============================================================
def save_samples(epoch_samples, epoch, output_dir, fs=16000) -> None:
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
def sample_generator(filepath, window_length=16384, fs=16000):
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
        LOGGER.error("Could not load {}: {}".format(filepath, str(e)))
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
    num_files = len(audio_path_list)
    num_valid = int(np.ceil(num_files * valid_ratio))
    num_test = int(np.ceil(num_files * test_ratio))
    num_train = num_files - num_valid - num_test

    if not (num_valid > 0 and num_test > 0 and num_train > 0):
        LOGGER.error("Please download DATASET '{}' and put it under current path !".format(DATASET_NAME))

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


# TODO: replace with torchaudio
# Adapted from https://github.com/caogang/wgan-gp/blob/master/gan_toy.py
def calc_gradient_penalty(netD, real_data, fake_data, batch_size, lmbda, device="cuda"):
    """
    Compute interpolation factors for WGAN-GP loss
    :param netD: Discriminators network
    :param real_data: Data comes fomrm dataset
    :param fake_data: Randomly generated fake data for discriminator
    :param batch_size: size(lenght) of batch
    :param lmbda: penalty coefficient
    :param device: use cuda if you want.
    :return: gradient penalty
    """
    alpha = torch.rand(batch_size, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)  # if device else alpha

    # Interpolate between real and fake data.
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    # if device:
    #     interpolates = interpolates.cuda()
    interpolates = interpolates.to(device)
    # interpolates = autograd.Variable(interpolates, requires_grad=True)
    interpolates.requires_grad = True

    # Evaluate discriminator
    disc_interpolates = netD(interpolates)

    # Obtain gradients of the discriminator with respect to the inputs
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if device else
                              torch.ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    # Compute MSE between 1.0 and the gradient of the norm penalty to make discriminator
    # to be a 1-Lipschitz function.
    gradient_penalty = lmbda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def numpy_to_var(numpy_data, device):
    """
    Convert numpy array to Variable.
    :param numpy_data: data in numpy array.
    :param device: use cuda if you want.
    :return: non gradient require torch.tensor
    """
    data = numpy_data[:, np.newaxis, :]
    data = torch.Tensor(data)
    # if cuda:
    #     data = data.cuda()
    data = data.to(device)
    data.requires_grad = False
    return data  # Variable(data, requires_grad=False)


# TODO: Implement tensorboard visualization
def plot_loss(D_cost_train, D_wass_train, D_cost_valid, D_wass_valid,
              G_cost, save_path) -> None:
    """
    Visualize Discriminators and Generator with respect to cost and Wasserstein(metric) loss using Matplotlib
    :param D_cost_train: Discriminators train cost
    :param D_wass_train: Discriminators train Wasserstein cost
    :param D_cost_valid: Discriminators validation cost
    :param D_wass_valid: Discriminators validation Wasserstein cost
    :param G_cost: Generator cost
    :param save_path: Image path. Save plot as image.
    :return: None
    """
    assert len(D_cost_train) == len(D_wass_train) == len(D_cost_valid) == len(D_wass_valid) == len(G_cost)

    save_path = os.path.join(save_path, "loss_curve.png")

    x = range(len(D_cost_train))

    y1 = D_cost_train
    y2 = D_wass_train
    y3 = D_cost_valid
    y4 = D_wass_valid
    y5 = G_cost

    plt.plot(x, y1, label='D_loss_train')
    plt.plot(x, y2, label='D_wass_train')
    plt.plot(x, y3, label='D_loss_valid')
    plt.plot(x, y4, label='D_wass_valid')
    plt.plot(x, y5, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path)


class Parameters:
    def __init__(self, konsol=False, args={}):
        self.args = {
            'num_epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'latent_dim': 100,
            'ngpus': 4,
            'model_size': 64,
            'output_dir': output,
            'epochs_per_sample': SAMPLE_EVERY,
            'lmbda': 10.0,
            'audio_dir': traindata,
            'phase-shuffle-shift-factor': 2,
            'post-proc-filt-len': 512,
            'lrelu-alpha': 0.2,
            'valid-ratio': 0.1,
            'test-ratio': 0.1,
            'learning-rate': 1e-4,
            'beta-one': 0.5,
            'beta-two': 0.9,
            'sample-size': SAMPLE_NUM
        }

        self.args = self.parse_arguments(konsol)

    def parse_arguments(self, cls=True):
        """
        Get command line arguments
        :param cls:
            if True: Parameters comming from konsol
            else: arguments coming from self.arguments
        :return: command line arguments with argparse package
        """
        parser = argparse.ArgumentParser(description='Train a WaveGAN on a given set of audio')

        parser.add_argument('-ms', '--model-size', dest='model_size', type=int, default=self.args['model_size'],
                            help='Model size parameter used in WaveGAN')
        parser.add_argument('-pssf', '--phase-shuffle-shift-factor', dest='shift_factor', type=int,
                            default=self.args['phase-shuffle-shift-factor'],
                            help='Maximum shift used by phase shuffle')
        parser.add_argument('-psb', '--phase-shuffle-batchwise', dest='batch_shuffle', action='store_true',
                            help='If true, apply phase shuffle to entire batches rather than individual samples')
        parser.add_argument('-ppfl', '--post-proc-filt-len', dest='post_proc_filt_len', type=int,
                            default=self.args['post-proc-filt-len'],
                            help='Length of post processing filter used by generator. Set to 0 to disable.')
        parser.add_argument('-lra', '--lrelu-alpha', dest='alpha', type=float, default=self.args['lrelu-alpha'],
                            help='Slope of negative part of LReLU used by discriminator')
        parser.add_argument('-vr', '--valid-ratio', dest='valid_ratio', type=float, default=self.args['valid-ratio'],
                            help='Ratio of audio files used for validation')
        parser.add_argument('-tr', '--test-ratio', dest='test_ratio', type=float, default=self.args['test-ratio'],
                            help='Ratio of audio files used for testing')
        parser.add_argument('-bs', '--batch-size', dest='batch_size', type=int, default=self.args['batch_size'],
                            help='Batch size used for training')
        parser.add_argument('-ne', '--num-epochs', dest='num_epochs', type=int, default=self.args['num_epochs'],
                            help='Number of epochs')
        parser.add_argument('-ng', '--ngpus', dest='ngpus', type=int, default=self.args['ngpus'],
                            help='Number of GPUs to use for training')
        parser.add_argument('-ld', '--latent-dim', dest='latent_dim', type=int, default=self.args['latent_dim'],
                            help='Size of latent dimension used by generator')
        parser.add_argument('-eps', '--epochs-per-sample', dest='epochs_per_sample', type=int,
                            default=self.args['epochs_per_sample'],
                            help='How many epochs between every set of samples generated for inspection')
        parser.add_argument('-ss', '--sample-size', dest='sample_size', type=int, default=self.args['sample-size'],
                            help='Number of inspection samples generated')
        parser.add_argument('-rf', '--regularization-factor', dest='lmbda', type=float, default=self.args['lmbda'],
                            help='Gradient penalty regularization factor')
        parser.add_argument('-lr', '--learning-rate', dest='learning_rate', type=float,
                            default=self.args['learning-rate'],
                            help='Initial ADAM learning rate')
        parser.add_argument('-bo', '--beta-one', dest='beta1', type=float, default=self.args['beta-one'],
                            help='beta_1 ADAM parameter')
        parser.add_argument('-bt', '--beta-two', dest='beta2', type=float, default=self.args['beta-two'],
                            help='beta_2 ADAM parameter')
        parser.add_argument('-v', '--verbose', dest='verbose', action='store_true')
        parser.add_argument('-audio_dir', '--audio_dir', dest='audio_dir', type=str, default=self.args['audio_dir'],
                            help='Path to directory containing audio files')
        parser.add_argument('-output_dir', '--output_dir', dest='output_dir', type=str, default=self.args['output_dir'],
                            help='Path to directory where model files will be output')
        if cls:
            args = parser.parse_args()
            args = vars(args)
        else:
            args = self.args

        return args

    def set_params(self):
        arguments = Parameters(False)
        arguments = arguments.args
        epochs = arguments['num_epochs']
        batch_size = arguments['batch_size']
        latent_dim = arguments['latent_dim']
        ngpus = arguments['ngpus']
        model_size = arguments['model_size']
        model_dir = make_path(os.path.join(arguments['output_dir'],
                                           datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
        arguments['model_dir'] = model_dir
        # save samples for every N epochs.
        epochs_per_sample = arguments['epochs_per_sample']
        # gradient penalty regularization factor.
        lmbda = arguments['lmbda']

        # Dir
        audio_dir = arguments['audio_dir']
        output_dir = arguments['output_dir']

        return epochs, batch_size, latent_dim, ngpus, model_size, model_dir, \
               epochs_per_sample, lmbda, audio_dir, output_dir

    def __call__(self, *args, **kwargs):
        """
        Returns parameters whenever call class
        :param args:
        :param kwargs:
        :return:
        """
        return self.args.keys()

    def __repr__(self):
        return str(self.args)
