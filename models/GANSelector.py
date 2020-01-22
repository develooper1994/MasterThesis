# TODO: It is an abstaction layer for different types of GAN
# standart module imports
import time
from typing import NoReturn

# classical machine learning imports

# other 3rd party libraries

# my modules
from models.DefaultTrainBuilder import RunBuilder, DefaultRunManager, DefaultTrainBuilder
# all trainers
from models.Trainers.DefaultTrainer import DefaultTrainer, epochs
# collect pair to "architectures" and import all models from "architectures"
from models.architectures.WaveGAN import WaveGAN


# utilities for all architectures

class GANSelector(DefaultTrainBuilder):
    # def __init__(self, netD, netG) -> None:
    def __init__(self, GAN, data_loader, epochs=1) -> None:
        super().__init__()
        self.m = DefaultRunManager()  # m indicates manager
        global gan_type
        self.epochs = epochs
        self.data_loader = data_loader
        # self.BATCH_NUM, self.train_iter, self.valid_iter, self.test_iter = self.dataloader.split_manage_data(arguments,
        #                                                                                                   batch_size)

        # select GAN
        if GAN == "wavegan":
            gan_type = WaveGAN()
            self.netD, self.netG = gan_type.netD, gan_type.netG  # WaveGANDiscriminator, WaveGANGenerator
            self.optimizerD, self.optimizerG = gan_type.optimizerD, gan_type.optimizerG  # wave_gan_utils.optimizers(arguments)

        # TODO add "segan" and "segan+(plus)" options
        else:
            print("I don't know your GAN. Make your own")
        self.set_nets(self.netD, self.netG)
        self.base_trainer = DefaultTrainer(self.netG, self.netD, self.optimizerG, self.optimizerD, gan_type,
                                           self.data_loader)

    def batches(self, **kwargs) -> NoReturn:
        self.base_trainer.train_gan_one_batch(kwargs)
        self.m.end_epoch()

    def all_epochs(self) -> NoReturn:
        start = time.time()
        for epoch in range(1, epochs + 1):
            self.m.begin_epoch()
            # one batch
            self.base_trainer.train_one_epoch(epoch, start)

    def train(self):
        # self.base_trainer.train()
        self.all_epochs()

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
