# standart modules
import logging

# my modules
# from utils.utils import time_since
from models.utils.BasicUtils import *


def init_console_logger(logger, verbose=False):
    stream_handler = logging.StreamHandler()
    if verbose:
        stream_handler.setLevel(logging.DEBUG)
    else:
        stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler("model.log")
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)


class file_logger:
    def __init__(self):
        self.LOGGER = logging.getLogger('wavegan')
        self.LOGGER.setLevel(logging.DEBUG)

        self.LOGGER.info('Initialized file_logger.')

    def start(self):
        init_console_logger(self.LOGGER)

    def training_info(self, start, D_cost_train_epoch_avg, D_wass_train_epoch_avg,
                      D_cost_valid_epoch_avg, D_wass_valid_epoch_avg, G_cost_epoch_avg):
        self.LOGGER.info("{} D_cost_train:{:.4f} | D_wass_train:{:.4f} | D_cost_valid:{:.4f} | D_wass_valid:{:.4f} | "
                         "G_cost:{:.4f}".format(time_since(start),
                                                D_cost_train_epoch_avg,
                                                D_wass_train_epoch_avg,
                                                D_cost_valid_epoch_avg,
                                                D_wass_valid_epoch_avg,
                                                G_cost_epoch_avg))

    def save_configurations(self):
        self.LOGGER.info('Saving configurations...')

    def loading_data(self):
        self.LOGGER.info('Loading audio data...')

    def start_training(self, epochs, batch_size, BATCH_NUM):
        self.LOGGER.info(
            'Starting training...EPOCHS={}, BATCH_SIZE={}, BATCH_NUM={}'.format(epochs, batch_size, BATCH_NUM))

    def epoch_info(self, start, epoch, epochs):
        self.LOGGER.info("{} Epoch: {}/{}".format(time_since(start), epoch, epochs))

    def batch_info(self, start, epoch, i, BATCH_NUM, D_cost_train, D_wass_train, G_cost):
        self.LOGGER.info(
            "{} Epoch={} Batch: {}/{} D_c:{:.4f} | D_w:{:.4f} | G:{:.4f}".format(time_since(start), epoch,
                                                                                 i, BATCH_NUM,
                                                                                 D_cost_train.cpu().data.numpy(),
                                                                                 D_wass_train.cpu().data.numpy(),
                                                                                 G_cost.data.cpu().numpy()))

    def batch_loss(self, start, D_cost_train_epoch_avg, D_wass_train_epoch_avg,
                   D_cost_valid_epoch_avg, D_wass_valid_epoch_avg, G_cost_epoch_avg):
        self.LOGGER.info("{} D_cost_train:{:.4f} | D_wass_train:{:.4f} | D_cost_valid:{:.4f} | D_wass_valid:{:.4f} | "
                         "G_cost:{:.4f}".format(time_since(start),
                                                D_cost_train_epoch_avg,
                                                D_wass_train_epoch_avg,
                                                D_cost_valid_epoch_avg,
                                                D_wass_valid_epoch_avg,
                                                G_cost_epoch_avg))

    def generating_samples(self):
        self.LOGGER.info("Generating samples...")

    def training_finished(self):
        self.LOGGER.info('>>>>>>>Training finished !<<<<<<<')

    def save_model(self):
        self.LOGGER.info("Saving models...")

    def save_loss_curve(self):
        self.LOGGER.info("Saving loss curve...")

    def end(self):
        self.LOGGER.info("All finished!")
