# from models.wavegan import WaveGAN
from typing import NoReturn

from config import params
from models.DefaultTrainBuilder import DefaultTrainBuilder


def main() -> NoReturn:
    # TODO: Implement for all gan types
    # wavegan = WaveGAN()
    # wavegan.train()

    # Dataset
    # dataloader = AudioDataset(input_dir=audio_dir, output_dir=output_dir)
    train_builder = DefaultTrainBuilder("wavegan-gp", None, epochs=1)
    multi_experiment = True  # False
    if multi_experiment:
        train_builder.train_experiments(params)  # x = torch.Size([1, 64, 250]) # fix 1 -> 150
    else:
        train_builder.train()  # x = torch.Size([150, 64, 250])


if __name__ == "__main__":
    print("Training Started")
    main()
    print("Training Ended")
