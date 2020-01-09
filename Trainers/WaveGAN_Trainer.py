import os
import time

import torch
from torch import autograd
from torch import optim
import json
from utils.utils import save_samples, Parameters, make_path, get_all_audio_filepaths, split_data, time_since, \
    numpy_to_var, calc_gradient_penalty, plot_loss, Parameters, SAMPLE_NUM
import numpy as np
import pprint
import pickle
import datetime
from models.wavegan import *
# from wavegan import *
from utils import *
from utils.logger import *

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")


class WaveGAN:
    def __init__(self):
        # =============Logger===============
        self.LOGGER = logging.getLogger('wavegan')
        self.LOGGER.setLevel(logging.DEBUG)

        self.LOGGER.info('Initialized logger.')
        init_console_logger(self.LOGGER)

        # =============Set Parameters===============
        arguments = Parameters(False)
        arguments = arguments.args
        self.epochs = arguments['num_epochs']
        self.batch_size = arguments['batch_size']
        self.latent_dim = arguments['latent_dim']
        self.ngpus = arguments['ngpus']
        self.model_size = arguments['model_size']
        self.model_dir = make_path(os.path.join(arguments['output_dir'],
                                                datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
        arguments['model_dir'] = self.model_dir
        # save samples for every N epochs.
        epochs_per_sample = arguments['epochs_per_sample']
        # gradient penalty regularization factor.
        self.lmbda = arguments['lmbda']

        # Dir
        self.audio_dir = arguments['audio_dir']
        self.output_dir = arguments['output_dir']

        self.netG = WaveGANGenerator(model_size=self.model_size, ngpus=self.ngpus,
                                     latent_dim=self.latent_dim, upsample=True)
        self.netD = WaveGANDiscriminator(model_size=self.model_size, ngpus=self.ngpus)

        self.netG = torch.nn.DataParallel(self.netG).to(device)
        self.netD = torch.nn.DataParallel(self.netD).to(device)

        # "Two time-scale update rule"(TTUR) to update netD 4x faster than netG.
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=arguments['learning-rate'],
                                     betas=(arguments['beta-one'], arguments['beta-two']))
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=arguments['learning-rate'],
                                     betas=(arguments['beta-one'], arguments['beta-two']))

        # Sample noise used for generated output.
        self.sample_noise = torch.randn(arguments['sample-size'], self.latent_dim)
        self.sample_noise = self.sample_noise.to(device)
        self.sample_noise.requires_grad = False  # sample_noise_Var = autograd.Variable(sample_noise, requires_grad=False)

        # Save config.
        self.LOGGER.info('Saving configurations...')
        config_path = os.path.join(self.model_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(arguments, f)

        # Load data.
        self.LOGGER.info('Loading audio data...')
        audio_paths = get_all_audio_filepaths(self.audio_dir)
        train_data, valid_data, test_data, train_size = split_data(audio_paths, arguments['valid-ratio'],
                                                                   arguments['test-ratio'],
                                                                   self.batch_size)
        self.TOTAL_TRAIN_SAMPLES = train_size
        self.BATCH_NUM = self.TOTAL_TRAIN_SAMPLES // self.batch_size

        self.train_iter = iter(train_data)
        self.valid_iter = iter(valid_data)
        self.test_iter = iter(test_data)

        self.history = []
        self.D_costs_train = []
        self.D_wasses_train = []
        self.D_costs_valid = []
        self.D_wasses_valid = []
        self.G_costs = []

    def train(self):
        # =============Train===============
        start = time.time()
        self.LOGGER.info('Starting training...EPOCHS={}, BATCH_SIZE={}, BATCH_NUM={}'.format(self.epochs, self.batch_size,
                                                                                        self.BATCH_NUM))
        for epoch in range(1, self.epochs + 1):
            self.LOGGER.info("{} Epoch: {}/{}".format(time_since(start), epoch, self.epochs))

            D_cost_train_epoch = []
            D_wass_train_epoch = []
            D_cost_valid_epoch = []
            D_wass_valid_epoch = []
            G_cost_epoch = []
            for i in range(1, self.BATCH_NUM + 1):
                # Set Discriminator parameters to require gradients.
                for p in self.netD.parameters():
                    p.requires_grad = True

                one = torch.tensor(1, dtype=torch.float)
                neg_one = torch.tensor(-1, dtype=torch.float)
                one = one.to(device)
                neg_one = neg_one.to(device)

                #############################
                # (1) Train Discriminator
                #############################
                for iter_dis in range(5):
                    self.netD.zero_grad()

                    # Noise
                    noise = torch.Tensor(self.batch_size, self.latent_dim).uniform_(-1, 1)
                    noise = noise.to(device)

                    noise.requires_grad = False  # noise_Var = Variable(noise, requires_grad=False)

                    real_data_Var = numpy_to_var(next(self.train_iter)['X'], device)

                    # a) compute loss contribution from real training data
                    D_real = self.netD(real_data_Var)
                    D_real = D_real.mean()  # avg loss
                    D_real.backward(neg_one)  # loss * -1

                    # b) compute loss contribution from generated data, then backprop.
                    fake = autograd.Variable(self.netG(noise).data)  # noise_Var
                    D_fake = self.netD(fake)
                    D_fake = D_fake.mean()
                    D_fake.backward(one)

                    # c) compute gradient penalty and backprop
                    gradient_penalty = calc_gradient_penalty(self.netD, real_data_Var.data,
                                                             fake.data, self.batch_size, self.lmbda,
                                                             device=device)
                    gradient_penalty.backward(one)

                    # Compute cost * Wassertein loss..
                    D_cost_train = D_fake - D_real + gradient_penalty
                    D_wass_train = D_real - D_fake

                    # Update gradient of discriminator.
                    self.optimizerD.step()

                    #############################
                    # (2) Compute Valid data
                    #############################
                    self.netD.zero_grad()

                    valid_data_Var = numpy_to_var(next(self.valid_iter)['X'], device)
                    D_real_valid = self.netD(valid_data_Var)
                    D_real_valid = D_real_valid.mean()  # avg loss

                    # b) compute loss contribution from generated data, then backprop.
                    fake_valid = self.netG(noise)  # noise_Var
                    D_fake_valid = self.netD(fake_valid)
                    D_fake_valid = D_fake_valid.mean()

                    # c) compute gradient penalty and backprop
                    gradient_penalty_valid = calc_gradient_penalty(self.netD, valid_data_Var.data,
                                                                   fake_valid.data, self.batch_size, self.lmbda,
                                                                   device=device)
                    # Compute metrics and record in batch history.
                    D_cost_valid = D_fake_valid - D_real_valid + gradient_penalty_valid
                    D_wass_valid = D_real_valid - D_fake_valid

                    if cuda:
                        D_cost_train = D_cost_train.cpu()
                        D_wass_train = D_wass_train.cpu()
                        D_cost_valid = D_cost_valid.cpu()
                        D_wass_valid = D_wass_valid.cpu()

                    # Record costs
                    D_cost_train_epoch.append(D_cost_train.data.numpy())
                    D_wass_train_epoch.append(D_wass_train.data.numpy())
                    D_cost_valid_epoch.append(D_cost_valid.data.numpy())
                    D_wass_valid_epoch.append(D_wass_valid.data.numpy())

                #############################
                # (3) Train Generator
                #############################
                # Prevent discriminator update.
                for p in self.netD.parameters():
                    p.requires_grad = False

                # Reset generator gradients
                self.netG.zero_grad()

                # Noise
                noise = torch.Tensor(self.batch_size, self.latent_dim).uniform_(-1, 1)
                # if cuda:
                #     noise = noise.cuda()
                noise = noise.to(device)
                noise.requires_grad = False  # noise_Var = Variable(noise, requires_grad=False)

                fake = self.netG(noise)  # noise_Var
                G = self.netD(fake)
                G = G.mean()

                # Update gradients.
                G.backward(neg_one)
                G_cost = -G

                self.optimizerG.step()

                # Record costs
                if cuda:
                    G_cost = G_cost.cpu()
                G_cost_epoch.append(G_cost.data.numpy())

                if i % (self.BATCH_NUM // 5) == 0:
                    self.LOGGER.info(
                        "{} Epoch={} Batch: {}/{} D_c:{:.4f} | D_w:{:.4f} | G:{:.4f}".format(time_since(start), epoch,
                                                                                             i, self.BATCH_NUM,
                                                                                             D_cost_train.data.numpy(),
                                                                                             D_wass_train.data.numpy(),
                                                                                             G_cost.data.numpy()))

            # Save the average cost of batches in every epoch.
            D_cost_train_epoch_avg = sum(D_cost_train_epoch) / float(len(D_cost_train_epoch))
            D_wass_train_epoch_avg = sum(D_wass_train_epoch) / float(len(D_wass_train_epoch))
            D_cost_valid_epoch_avg = sum(D_cost_valid_epoch) / float(len(D_cost_valid_epoch))
            D_wass_valid_epoch_avg = sum(D_wass_valid_epoch) / float(len(D_wass_valid_epoch))
            G_cost_epoch_avg = sum(G_cost_epoch) / float(len(G_cost_epoch))

            self.D_costs_train.append(D_cost_train_epoch_avg)
            self.D_wasses_train.append(D_wass_train_epoch_avg)
            self.D_costs_valid.append(D_cost_valid_epoch_avg)
            self.D_wasses_valid.append(D_wass_valid_epoch_avg)
            self. G_costs.append(G_cost_epoch_avg)

            self.LOGGER.info("{} D_cost_train:{:.4f} | D_wass_train:{:.4f} | D_cost_valid:{:.4f} | D_wass_valid:{:.4f} | "
                        "G_cost:{:.4f}".format(time_since(start),
                                               D_cost_train_epoch_avg,
                                               D_wass_train_epoch_avg,
                                               D_cost_valid_epoch_avg,
                                               D_wass_valid_epoch_avg,
                                               G_cost_epoch_avg))

            # Generate audio samples.
            if epoch % self.epochs_per_sample == 0:
                self.LOGGER.info("Generating samples...")
                sample_out = self.netG(self.sample_noise)  # sample_noise_Var
                if cuda:
                    sample_out = sample_out.cpu()
                sample_out = sample_out.data.numpy()
                save_samples(sample_out, epoch, self.output_dir)

            # TODO: Early stopping by Inception Score(IS)

        self.LOGGER.info('>>>>>>>Training finished !<<<<<<<')

        # TODO: Implement check point and load from checkpoint
        # Save model
        self.LOGGER.info("Saving models...")
        netD_path = os.path.join(self.output_dir, "discriminator.pkl")
        netG_path = os.path.join(self.output_dir, "generator.pkl")
        torch.save(self.netD.state_dict(), netD_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        torch.save(self.netG.state_dict(), netG_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)

        # Plot loss curve.
        self.LOGGER.info("Saving loss curve...")
        plot_loss(self.D_costs_train, self.D_wasses_train,
                  self.D_costs_valid, self.D_wasses_valid, self.G_costs, self.output_dir)

        self.LOGGER.info("All finished!")

    # def __call__(self, *args, **kwargs):
    #     self.train()


class WaveGAN_Trainer:
    def __init__(self):
        pass
