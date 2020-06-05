import torch

from not_functional.config import noise_latent_dim
from not_functional.models.architectures import WaveGANDiscriminator
from not_functional.models.architectures.Generators import WaveGANGenerator

# TODO: write document for each function.
from not_functional.models.utils.BaseGANUtils import BaseGANUtils


class WaveGANUtils(BaseGANUtils):
    def __init__(self):
        super().__init__(generator=WaveGANGenerator, discriminator=WaveGANDiscriminator)

    def create_network(self, **kwargs):
        return super().create_network(**kwargs)

    def optimizers(self, arguments, **networks):
        return super().optimizers(arguments, **networks)

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