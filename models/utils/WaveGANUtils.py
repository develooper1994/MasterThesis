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

    def create_network(self, model_size, ngpus, latent_dim):
        return super().create_network(model_size, ngpus, latent_dim)

    def optimizers(self, arguments):
        return super().optimizers(arguments)

    def sample_noise(self, arguments, latent_dim, device):
        return super().sample_noise(arguments, latent_dim, device)

    def generate_audio_samples(self, Logger, sample_noise, epoch, output_dir):
        return super().generate_audio_samples(Logger, sample_noise, epoch, output_dir)
