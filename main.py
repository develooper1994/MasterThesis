# from models.wavegan import WaveGAN
from models.architectures.wavegan import WaveGAN

from models.Trainers.Trainer import Trainer


def main():
    # TODO: Implement for all gan types
    wavegan = WaveGAN()
    wavegan.train()

    # trainer = Trainer("wavegan")
    # trainer.train()


if __name__ == "__main__":
    print("Training Started")
    main()
    print("Training Ended")