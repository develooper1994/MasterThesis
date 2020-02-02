from collections import OrderedDict, namedtuple
from itertools import product  # cartesian product
from typing import List, Any

import torch
import random
import numpy as np
import logging
import os

# EmTraining
from torch import device

EPOCHS = 1  # 1 # 10 # 180
# NOTE!!!: Bigger BATCH_SIZE; faster training and slower gradient degradation

SAMPLE_EVERY = 1  # Generate audio samples every 1 epoch.

# Model(Network)
MODEL = "wavegan-gp"
# if you want to change optimizers look at models.optimizers.BaseOptimizer.py

#############################
# DataSet Path
#############################
target_signals_dir: str = '/home/selcuk/.pytorch/piano'  # 'sc09/'
#############################
# Model Params
#############################
model_prefix: str = 'exp1'  # name of the model to be saved
n_iterations: int = 10000
lr_g = 1e-4  # generator network.
lr_d = 1e-4  # discriminator and decoder networks.
lr_e = 1e-4  # encoder for autoencoder network. NOT USED YET!
beta1: int = 0  # generally "0.9" but for now "0"
beta2: float = 0.999
decay_lr = False  # used to linearly deay learning rate untill reaching 0 at iteration 100,000
generator_batch_size_factor: int = 1  # in some cases we might try to update the generator with double batch size used in the discriminator https://arxiv.org/abs/1706.08500
n_critic: int = 5  # update generator every n_critic steps
# gradient penalty regularization factor.
p_coeff: int = 10
batch_size: int = 200  # 193 # 190 # 175 # 150 # 100 # 80 # 64 # 10
noise_latent_dim: int = 100  # size of the sampling noise
model_capacity_size: int = 64  # model capacity during training can be reduced to 32 for larger window length of 2 seconds and 4 seconds
# rate of storing validation and costs params
progress_bar_step_iter_size: int = batch_size  # 200
store_cost_every: int = progress_bar_step_iter_size  # 300
#############################
# Backup Params
#############################
take_backup = True
backup_every_n_iters: int = 1000
# it is also input size of discriminator
save_samples_every: int = 1000  # Generate 10 samples every sample generation.
output_dir = 'output/'  # '../../DataSet/output'
if not (os.path.isdir(output_dir)):
    os.makedirs(output_dir)
#############################
# Audio Reading Params
#############################
window_length: int = 16384  # [16384, 32768, 65536] in case of a longer window change model_capacity_size to 32
# signal processing configurations.
# NOTE: DON'T CHANGE IF YOU DON'T REALLLLYYYY NEED
sampling_rate: int = 16000
normalize_audio = True
num_channels: int = 1

#############################
# Logger init
#############################
LOGGER = logging.getLogger(MODEL)
LOGGER.setLevel(logging.DEBUG)
#############################
# Torch Init and seed setting
#############################
cuda: bool = torch.cuda.is_available()
device: device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
# update the seed
manual_seed: int = 2020  # 2019

# Hyper parameter experiments.
# order is important. DON'T CHANGE
params = OrderedDict(
    # epochs=[EPOCHS],
    # lr=[.01, .001],
    #############################
    # DataSet Path
    #############################
    target_signals_dir=[target_signals_dir],
    #############################
    # Model Params
    #############################
    model_prefix=[model_prefix],
    n_iterations=[n_iterations],
    lr_g=[lr_g],
    lr_d=[lr_d],
    lr_e=[lr_e],
    beta1=[beta1],
    beta2=[beta2],
    decay_lr=[decay_lr],
    generator_batch_size_factor=[generator_batch_size_factor],
    n_critic=[n_critic],
    p_coeff=[p_coeff],
    batch_size=[batch_size],
    noise_latent_dim=[noise_latent_dim],
    model_capacity_size=[model_capacity_size],
    store_cost_every=[store_cost_every],
    progress_bar_step_iter_size=[progress_bar_step_iter_size],
    #############################
    # Backup Params
    #############################
    take_backup=[take_backup],
    backup_every_n_iters=[backup_every_n_iters],
    save_samples_every=[save_samples_every],
    output_dir=[output_dir],
    #############################
    # Audio Reading Params
    #############################
    window_length=[window_length],
    sampling_rate=[sampling_rate],
    normalize_audio=[normalize_audio],
    num_channels=[num_channels],
    #############################
    # Torch Init and seed setting
    #############################
    cuda=[cuda],
    device=[device],
    manual_seed=[manual_seed],  # update the seed

    shuffle=[True, False],
    MODEL=[MODEL],  # Last element
)


# Don't delete extremly powerful technique.
class RunBuilder:
    """
    Takes all parameters with variable names and values and insert into a list
    """

    @staticmethod
    def get_runs(params) -> List[Any]:
        """
        Takes parameters and makes first it dictionary.
        Second appends all values or keys of variable in to a list.
        @param params: takes an OrderedDict to try all experiments in one loop and also to show results on any report.
        @return: list of values or keys of variable
        """
        Run = namedtuple('Run', params.keys())

        return [Run(*v) for v in product(*params.values())]


Runs = RunBuilder.get_runs(params)

random.seed(manual_seed)
torch.manual_seed(manual_seed)
np.random.seed(manual_seed)
if cuda:
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.empty_cache()
