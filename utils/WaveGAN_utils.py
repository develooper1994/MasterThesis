import os
import json
import logging

import torch
from torch import optim

from models.wavegan import WaveGANGenerator, WaveGANDiscriminator

# TODO: write document for each function.
# from utils.utils import Logger

# File Logger Configfuration
LOGGER = logging.getLogger('wavegan')
LOGGER.setLevel(logging.DEBUG)


def parallel_models(device, *nets):
    net = []
    for n in nets:
        n = torch.nn.DataParallel(n).to(device)
        net.append(n)
    return net


def create_network(model_size, ngpus, latent_dim, device):
    netG = WaveGANGenerator(model_size=model_size, ngpus=ngpus,
                            latent_dim=latent_dim, upsample=True)
    netD = WaveGANDiscriminator(model_size=model_size, ngpus=ngpus)

    netG, netD = parallel_models(device, netG, netD)
    return netG, netD


def optimizers(netG, netD, arguments):
    optimizerG = optim.Adam(netG.parameters(), lr=arguments['learning-rate'],
                            betas=(arguments['beta-one'], arguments['beta-two']))
    optimizerD = optim.Adam(netD.parameters(), lr=arguments['learning-rate'],
                            betas=(arguments['beta-one'], arguments['beta-two']))
    return optimizerG, optimizerD


def sample_noise(arguments, latent_dim, device):
    sample_noise = torch.randn(arguments['sample-size'], latent_dim)
    sample_noise = sample_noise.to(device)
    sample_noise.requires_grad = False  # sample_noise_Var = autograd.Variable(sample_noise, requires_grad=False)
    return sample_noise


def creat_dump(model_dir, arguments):
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(arguments, f)


def split_manage_data(audio_dir, arguments, batch_size):
    from utils.utils import get_all_audio_filepaths
    audio_paths = get_all_audio_filepaths(audio_dir)
    from utils.utils import split_data
    train_data, valid_data, test_data, train_size = split_data(audio_paths, arguments['valid-ratio'],
                                                               arguments['test-ratio'],
                                                               batch_size)
    TOTAL_TRAIN_SAMPLES = train_size
    BATCH_NUM = TOTAL_TRAIN_SAMPLES // batch_size

    train_iter, valid_iter, test_iter = iter(train_data), iter(valid_data), iter(test_data)

    return BATCH_NUM, train_iter, valid_iter, test_iter


def tocpu_all(D_cost_train, D_wass_train, D_cost_valid, D_wass_valid):
    D_cost_train = D_cost_train.cpu()
    D_wass_train = D_wass_train.cpu()
    D_cost_valid = D_cost_valid.cpu()
    D_wass_valid = D_wass_valid.cpu()
    return D_cost_train, D_wass_train, D_cost_valid, D_wass_valid


def tocuda_all(D_cost_train, D_wass_train, D_cost_valid, D_wass_valid):
    D_cost_train = D_cost_train.cuda()
    D_wass_train = D_wass_train.cuda()
    D_cost_valid = D_cost_valid.cuda()
    D_wass_valid = D_wass_valid.cuda()
    return D_cost_train, D_wass_train, D_cost_valid, D_wass_valid


def record_costs(D_cost_train_epoch, D_wass_train_epoch, D_cost_valid_epoch, D_wass_valid_epoch,
                 D_cost_train, D_wass_train, D_cost_valid, D_wass_valid):
    D_cost_train_epoch.append(D_cost_train.data.numpy())
    D_wass_train_epoch.append(D_wass_train.data.numpy())
    D_cost_valid_epoch.append(D_cost_valid.data.numpy())
    D_wass_valid_epoch.append(D_wass_valid.data.numpy())


def compute_and_record_batch_history(D_fake_valid, D_real_valid, D_cost_train, D_wass_train, gradient_penalty_valid,
                                     D_cost_train_epoch, D_wass_train_epoch, D_cost_valid_epoch, D_wass_valid_epoch):
    D_cost_valid = D_fake_valid - D_real_valid + gradient_penalty_valid
    D_wass_valid = D_real_valid - D_fake_valid

    D_cost_train, D_wass_train, D_cost_valid, D_wass_valid = \
        tocpu_all(D_cost_train, D_wass_train, D_cost_valid, D_wass_valid)

    # Record costs
    record_costs(D_cost_train_epoch, D_wass_train_epoch, D_cost_valid_epoch, D_wass_valid_epoch,
                 D_cost_train, D_wass_train, D_cost_valid, D_wass_valid)


def save_avg_cost_one_epoch(D_cost_train_epoch, D_wass_train_epoch, D_cost_valid_epoch, D_wass_valid_epoch,
                            G_cost_epoch,
                            D_costs_train, D_wasses_train, D_costs_valid, D_wasses_valid, G_costs, Logger, start):
    # Save the average cost of batches in every epoch.
    D_cost_train_epoch_avg = sum(D_cost_train_epoch) / float(len(D_cost_train_epoch))
    D_wass_train_epoch_avg = sum(D_wass_train_epoch) / float(len(D_wass_train_epoch))
    D_cost_valid_epoch_avg = sum(D_cost_valid_epoch) / float(len(D_cost_valid_epoch))
    D_wass_valid_epoch_avg = sum(D_wass_valid_epoch) / float(len(D_wass_valid_epoch))
    G_cost_epoch_avg = sum(G_cost_epoch) / float(len(G_cost_epoch))

    D_costs_train.append(D_cost_train_epoch_avg)
    D_wasses_train.append(D_wass_train_epoch_avg)
    D_costs_valid.append(D_cost_valid_epoch_avg)
    D_wasses_valid.append(D_wass_valid_epoch_avg)
    G_costs.append(G_cost_epoch_avg)

    Logger.batch_loss(start, D_cost_train_epoch_avg, D_wass_train_epoch_avg,
                      D_cost_valid_epoch_avg, D_wass_valid_epoch_avg, G_cost_epoch_avg)


def generate_audio_samples(Logger, netG, sample_noise, epoch, output_dir):
    from utils.utils import save_samples

    Logger.generating_samples()

    sample_out = netG(sample_noise)  # sample_noise_Var
    sample_out = sample_out.cpu().data.numpy()
    save_samples(sample_out, epoch, output_dir)
