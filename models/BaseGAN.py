# TODO: It is an abstaction layer for different types of GAN
class BaseDiscriminator:
    pass


class BaseGenerator:
    pass


class BaseGAN:
    def __init__(self, netD, netG) -> None:
        self.set_GAN(netD, netG)

    def set_discriminator(self, netD):
        self.netD = netD

    def get_discriminator(self):
        return self.netD

    def set_generator(self, netG):
        self.netG = netG

    def get_generator(self):
        return self.netG

    def set_GAN(self, netD, netG):
        self.set_generator(netD)
        self.set_generator(netG)

    def get_GAN(self):
        return self.get_discriminator(), self.get_generator()
