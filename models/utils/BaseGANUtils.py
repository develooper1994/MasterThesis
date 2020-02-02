import torch

from models.optimizers.BaseOptimizer import optimizers
# File Logger Configfuration
from models.utils.BasicUtils import parallel_models


# TODO: write document for each function.


class BaseGANUtils:
    # def __init__(self, generator, discriminator):
    # self.generator = generator
    # self.discriminator = discriminator
    def __init__(self, **networks):
        self.networks = networks
        self.networks_names = list(networks.keys())
        self.network_values = list(networks.values())

    def create_network(self, **kwargs):
        # self.generator = self.generator(upsample=True, **kwargs)
        # self.discriminator = self.discriminator(kwargs)
        #
        # netG, netD = parallel_models(self.generator, self.discriminator)
        # return netG, netD
        return [
            parallel_models(net(**kwargs))[0]
            if net_name.lower() == "discriminator"
            else parallel_models(net(upsample=True, **kwargs))[0]
            for net_name, net in zip(self.networks_names, self.network_values)
        ]

    def optimizers(self, arguments, **networks):
        return optimizers(arguments, **networks)

    def sample_noise(self, arguments, latent_dim, device):
        sample_noise = torch.randn(arguments['sample-size'], latent_dim)
        sample_noise = sample_noise.to(device)
        sample_noise.requires_grad = False  # sample_noise_Var = autograd.Variable(sample_noise, requires_grad=False)
        return sample_noise

    def generate_audio_samples(self, Logger, sample_noise, epoch, output_dir):
        from models.DataLoader.AudioDataset import AudioDataset

        Logger.generating_samples()

        sample_out = self.generator(sample_noise)  # sample_noise_Var
        sample_out = sample_out.cpu().data.numpy()
        # from models.DataLoader.custom_DataLoader import save_samples
        # save_samples(sample_out, epoch, output_dir)
        dataset = AudioDataset(output_dir=output_dir)
        dataset.save_samples(epoch, sample_out)
