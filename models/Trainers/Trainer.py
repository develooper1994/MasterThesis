# That modules connects BaseTrainer.py with GANSelector.py to make compact Trainer mechanism

# my modules
import torch
from torch.utils.data import dataset, DataLoader

from models.Trainers.BaseTrainer import BaseTrainer
from models.GANSelector import GANSelector
from config import DATASET_NAME, DATASET_NAME, OUTPUT_PATH, SAMPLE_NUM, WINDOW_LENGHT, FS, EPOCHS, BATCH_SIZE, MODEL

from models.utils.WaveGANUtils import WaveGANUtils
from models.DataLoader.AudioDataset import AudioDataset
from models.utils.BasicUtils import Parameters


class Trainer:
    def __init__(self, GAN_name: str = MODEL) -> None:
        self.GAN_name = GAN_name.lower()
        self.GAN = GANSelector(self.GAN_name)

        arguments = Parameters(False)
        self.arguments = arguments.args
        self.dataset = AudioDataset(input_dir=DATASET_NAME, output_dir=OUTPUT_PATH)
        self.dataloader = self.dataset.split_manage_data(self.arguments, BATCH_SIZE)

        # self.select_trainer(self.GAN_name)
        self.trainer = BaseTrainer(self.GAN.netG, self.GAN.netD, self.GAN.optimizerG, self.GAN.optimizerD,
                              use_cuda=torch.cuda.is_available())

    def train(self):
        self.trainer.train(self.dataloader, EPOCHS, save_training_gif=True)
