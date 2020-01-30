import torch

from config import noise_latent_dim
from models.Discriminators.WaveGANDiscriminator import WaveGANDiscriminator
from models.Generators.WaveGANGenerator import WaveGANGenerator

# TODO: write document for each function.
# from models.utils.utils import Logger
from models.utils.BaseGANUtils import BaseGANUtils

# File Logger Configfuration

# LOGGER = logging.getLogger('wavegan')
# LOGGER.setLevel(logging.DEBUG)


class WaveGANUtils(BaseGANUtils):
    def __init__(self):
        super().__init__(WaveGANGenerator, WaveGANDiscriminator)

    def create_network(self, model_size):
        return super().create_network(model_size)

    def optimizers(self, arguments):
        return super().optimizers(arguments)

    def sample_noise(self, arguments, latent_dim, device):
        return super().sample_noise(arguments, latent_dim, device)

    def generate_audio_samples(self, Logger, sample_noise, epoch, output_dir):
        return super().generate_audio_samples(Logger, sample_noise, epoch, output_dir)


if __name__ == '__main__':
    for slice_len in [16384, 32768, 65536]:
        G = WaveGANGenerator(verbose=True, upsample=True, use_batch_norm=True, slice_len=slice_len)
        out = G(torch.randn(10, noise_latent_dim))
        print(out.shape)
        assert (out.shape == (10, 1, slice_len))
        print('==========================')

        D = WaveGANDiscriminator(verbose=True, use_batch_norm=True, slice_len=slice_len)
        out2 = D(torch.randn(10, 1, slice_len))
        print(out2.shape)
        assert (out2.shape == (10, 1))
        print('==========================')