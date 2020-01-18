# TODO: It is an abstaction layer for different types of GAN
# collect pair to "architectures" and import all models from "architectures"
from models.Discriminators import WaveGANDiscriminator
from models.Generators import WaveGANGenerator

# utilities for all architectures
from models.utils.WaveGANUtils import WaveGANUtils


class GANSelector:
    # def __init__(self, netD, netG) -> None:
    def __init__(self, GAN, arguments) -> None:
        # select GAN
        if GAN == "wavegan":
            self.netD, self.netG = WaveGANDiscriminator, WaveGANGenerator
            WaveGANUtil = WaveGANUtils()
            self.optimizerD, self.optimizerG = WaveGANUtil.optimizers(arguments)


        # TODO add "segan" and "segan+(plus)" options
        else:
            print("I don't know your GAN. Make your own")
        self.set_nets(self.netD, self.netG)

    def set_discriminator(self, netD):
        self.netD = netD

    def get_discriminator(self):
        return self.netD

    def set_generator(self, netG):
        self.netG = netG

    def get_generator(self):
        return self.netG

    def set_nets(self, netD, netG):
        self.set_generator(netD)
        self.set_generator(netG)

    def get_nets(self):
        return self.get_discriminator(), self.get_generator()
