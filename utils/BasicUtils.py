# standart library
import argparse
import datetime
import math
import os
import pickle
import time

import numpy as np
# 3'rd party
import torch
# my libraries
from torch import autograd

from config import DATASET_NAME, OUTPUT_PATH, EPOCHS, BATCH_SIZE, SAMPLE_EVERY, SAMPLE_NUM

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")
print("Training device: {}".format(device))


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


def prevent_net_update(net):
    for p in net.parameters():
        p.requires_grad = False


def require_net_update(net):
    for p in net.parameters():
        p.requires_grad = True


def numpy_to_var(numpy_data, device):
    """
    Convert numpy array to Variable.
    :param numpy_data: data in numpy array.
    :param device: use cuda if you want.
    :return: non gradient require torch.tensor
    """
    data = numpy_data[:, np.newaxis, :]
    data = torch.Tensor(data)
    data = data.to(device)
    data.requires_grad = False
    return data


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
    interpolates = interpolates.to(device)
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


def save_models(output_dir, netD, netG):
    netD_path = os.path.join(output_dir, "discriminator.pkl")
    netG_path = os.path.join(output_dir, "generator.pkl")
    torch.save(netD.state_dict(), netD_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(netG.state_dict(), netG_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)