# from models.wavegan import WaveGAN
from typing import NoReturn

from config import params, MODEL
from models.DefaultTrainBuilder import DefaultTrainBuilder


def main() -> NoReturn:
    train_builder = DefaultTrainBuilder(MODEL, None, epochs=1)
    multi_experiment = True  # False
    if multi_experiment:
        train_builder.train_experiments()  # x = torch.Size([1, 64, 250]) # fix 1 -> 150
    else:
        train_builder.train()  # x = torch.Size([150, 64, 250])


if __name__ == "__main__":
    print("Training Started")
    main()
    print("Training Ended")
