# modules
from models.wavegan import WaveGAN

import torch


# cuda = True if torch.cuda.is_available() else False
# device = torch.device("cuda" if cuda else "cpu")


class Trainer:
    def select_trainer(self, trainer: str = None) -> None:
        trainer = trainer.lower()
        if trainer is None:
            trainer = self.GAN_name

        # select GAN
        if trainer == "wavegan":
            GAN = WaveGAN()
            GAN.train()
        elif trainer == "segan":
            pass
        elif trainer == "segan+" or "seganplus":
            pass
        else:
            print("I don't know your GAN")

    def __init__(self, GAN_name: str) -> None:
        self.GAN_name = GAN_name
        self.select_trainer(GAN_name)

    def __call__(self, *args, **kwargs) -> None:
        self.select_trainer()
