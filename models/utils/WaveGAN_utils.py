import logging

import torch
from torch import optim

from models.Discriminators.WaveGAN_Discriminator import WaveGANDiscriminator
from models.Generators.WaveGAN_Generator import WaveGANGenerator

# TODO: write document for each function.
# from models.utils.utils import Logger
from models.utils.BaseGAN_Utils import BaseGAN_utils

# File Logger Configfuration
from models.utils.BasicUtils import parallel_models

LOGGER = logging.getLogger('wavegan')
LOGGER.setLevel(logging.DEBUG)


class WaveGAN_utils(BaseGAN_utils):
    def __init__(self):
        super().__init__(WaveGANGenerator, WaveGANDiscriminator)

    def create_network(self, model_size, ngpus, latent_dim, device):
        return super().create_network(model_size, ngpus, latent_dim, device)

    def optimizers(self, arguments):
        return super().optimizers(arguments)

    def sample_noise(self, arguments, latent_dim, device):
        return super().sample_noise(arguments, latent_dim, device)

    def generate_audio_samples(self, Logger, sample_noise, epoch, output_dir):
        return super().generate_audio_samples(Logger, sample_noise, epoch, output_dir)


def create_network(model_size, ngpus, latent_dim, device):
    netG = WaveGANGenerator(model_size=model_size, ngpus=ngpus,
                            latent_dim=latent_dim, upsample=True)
    netD = WaveGANDiscriminator(model_size=model_size, ngpus=ngpus)

    netG, netD = parallel_models(device, netG, netD)
    return netG, netD


def optimizers(netG, netD, arguments):
    optimizerG = optim.Adam(netG.parameters(), lr=arguments['learning-rate'],
                            betas=(arguments['beta-one'], arguments['beta-two']))
    optimizerD = optim.Adam(netD.parameters(), lr=arguments['learning-rate'],
                            betas=(arguments['beta-one'], arguments['beta-two']))
    return optimizerG, optimizerD


def generate_audio_samples(Logger, netG, sample_noise, epoch, output_dir):
    from models.custom_DataLoader.custom_DataLoader import save_samples

    Logger.generating_samples()

    sample_out = netG(sample_noise)  # sample_noise_Var
    sample_out = sample_out.cpu().data.numpy()
    save_samples(sample_out, epoch, output_dir)


def sample_noise(arguments, latent_dim, device):
    sample_noise = torch.randn(arguments['sample-size'], latent_dim)
    sample_noise_device = sample_noise.to(device)
    sample_noise_device.requires_grad = False  # sample_noise_Var = autograd.Variable(sample_noise, requires_grad=False)
    return sample_noise_device