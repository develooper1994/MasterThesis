import logging

import torch
from torch import optim

from models.Discriminators.WaveGAN_Discriminator import WaveGANDiscriminator
from models.Generators.WaveGAN_Generator import WaveGANGenerator

# TODO: write document for each function.
# from models.utils.utils import Logger
from models.utils.BaseGAN_Utils import BaseGANUtils

# File Logger Configfuration
from models.utils.BasicUtils import parallel_models

LOGGER = logging.getLogger('wavegan')
LOGGER.setLevel(logging.DEBUG)


class WaveGAN_utils(BaseGANUtils):
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
