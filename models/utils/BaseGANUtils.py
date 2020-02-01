import torch

from models.optimizers.BaseOptimizer import optimizers
# File Logger Configfuration
from models.utils.BasicUtils import parallel_models


# TODO: write document for each function.


class BaseGANUtils:
    def __init__(self, Generator, Discriminator):
        self.netG = Generator
        self.netD = Discriminator

    def create_network(self, model_size):
        self.netG = self.netG(model_size=model_size, upsample=True)
        self.netD = self.netD(model_size=model_size)

        netG, netD = parallel_models(self.netG, self.netD)
        return netG, netD

    def optimizers(self, arguments):
        return optimizers(self.netD, self.netG, arguments)

    def sample_noise(self, arguments, latent_dim, device):
        sample_noise = torch.randn(arguments['sample-size'], latent_dim)
        sample_noise = sample_noise.to(device)
        sample_noise.requires_grad = False  # sample_noise_Var = autograd.Variable(sample_noise, requires_grad=False)
        return sample_noise

    def generate_audio_samples(self, Logger, sample_noise, epoch, output_dir):
        from models.DataLoader.AudioDataset import AudioDataset

        Logger.generating_samples()

        sample_out = self.netG(sample_noise)  # sample_noise_Var
        sample_out = sample_out.cpu().data.numpy()
        # from models.DataLoader.custom_DataLoader import save_samples
        # save_samples(sample_out, epoch, output_dir)
        dataset = AudioDataset(output_dir=output_dir)
        dataset.save_samples(epoch, sample_out)
