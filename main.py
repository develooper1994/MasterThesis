# from models.wavegan import WaveGAN
from typing import NoReturn

from models.DataLoader.AudioDataset import AudioDataset
from models.GANSelector import GANSelector
from models.Trainers.DefaultTrainer import audio_dir, output_dir


def main() -> NoReturn:
    # TODO: Implement for all gan types
    # wavegan = WaveGAN()
    # wavegan.train()

    # Dataset
    dataloader = AudioDataset(input_dir=audio_dir, output_dir=output_dir)
    gan_selector = GANSelector("wavegan", dataloader, epochs=1)
    gan_selector.train()


if __name__ == "__main__":
    print("Training Started")
    main()
    print("Training Ended")
