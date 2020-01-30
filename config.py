from collections import OrderedDict

import torch
import random
import numpy as np
import logging
import os

# EmTraining
EPOCHS = 1  # 1 # 10 # 180
# NOTE!!!: Bigger BATCH_SIZE; faster training and slower gradient degradation
BATCH_SIZE = 150  # 193 # 190 # 175 # 150 # 100 # 80 # 64 # 10

# signal processing configurations.
# NOTE: DON'T CHANGE IF YOU DON'T REALLLLYYYY NEED
WINDOW_LENGHT = 16384
FS = 16000

SAMPLE_EVERY = 1  # Generate audio samples every 1 epoch.
# it is also input size of discriminator
SAMPLE_NUM = int(FS/16)  # Generate 10 samples every sample generation.

# Data
DATASET_NAME = "/home/selcuk/.pytorch/sc09/"  # 'sc09/'
OUTPUT_PATH = "output/"

# Model(Network)
MODEL = "wavegan"
# if you want to change optimizers look at models.optimizers.BaseOptimizer.py

#############################
# DataSet Path
#############################s
target_signals_dir: str = '/home/selcuk/.pytorch/piano'
#############################
# Model Params
#############################
model_prefix: str = 'exp1'  # name of the model to be saved
n_iterations: int = 10000
lr_g = 1e-4
lr_d = 1e-4
beta1: int = 0
beta2: float = 0.999
decay_lr = False  # used to linearly deay learning rate untill reaching 0 at iteration 100,000
generator_batch_size_factor: int = 1  # in some cases we might try to update the generator with double batch size used in the discriminator https://arxiv.org/abs/1706.08500
n_critic: int = 5  # update generator every n_critic steps
# gradient penalty regularization factor.
p_coeff: int = 10
batch_size: int = 150  # 10
noise_latent_dim: int = 100  # size of the sampling noise
model_capacity_size: int = 64  # model capacity during training can be reduced to 32 for larger window length of 2 seconds and 4 seconds
# rate of storing validation and costs params
store_cost_every: int = 300
progress_bar_step_iter_size: int = 200
#############################
# Backup Params
#############################
take_backup = True
backup_every_n_iters: int = 1000
save_samples_every: int = 1000
output_dir = '../../DataSet/output'
if not (os.path.isdir(output_dir)):
    os.makedirs(output_dir)
#############################
# Audio Reading Params
#############################
window_length: int = 16384  # [16384, 32768, 65536] in case of a longer window change model_capacity_size to 32
sampling_rate: int = 16000
normalize_audio = True
num_channels: int = 1

#############################
# Logger init
#############################
LOGGER = logging.getLogger('wavegan')
LOGGER.setLevel(logging.DEBUG)
#############################
# Torch Init and seed setting
#############################
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
# update the seed
manual_seed: int = 2020  # 2019
random.seed(manual_seed)
torch.manual_seed(manual_seed)
np.random.seed(manual_seed)
if cuda:
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.empty_cache()


# experiments
params = OrderedDict(
        # epochs=[EPOCHS],
        lr=[.01, .001],
        batch_size=[BATCH_SIZE, 100, 1000],
        shuffle=[True, False],
        # dataset_name=[DATASET_NAME],
        # output_name=[OUTPUT_PATH],
        # window_lenght=[WINDOW_LENGHT],
        # sample_every=[SAMPLE_EVERY],
        # sample_num=[SAMPLE_NUM],
        # Model(Network)
        MODEL=["wavegan"],
    )