# from models.wavegan import WaveGAN
from models.Trainers.Trainer import Trainer


def main():
    # TODO: Implement for all gan types
    # gan = WaveGAN()  # default behaviour is train.
    # gan.train()
    Trainer("wavegan")


if __name__ == "__main__":
    print("Training Started")
    main()
    print("Training Ended")