# Standart library
import time

# torch imports
import torch
from torch import autograd

from models.DataLoader.AudioDataset import AudioDataset
# my modules
from models.losses.BaseLoss import wassertein_loss
from models.utils import BasicUtils as utls_basic
from models.utils.BasicUtils import get_params, require_net_update, numpy_to_var, calc_gradient_penalty, \
    prevent_net_update, cuda, creat_dump, compute_and_record_batch_history, save_avg_cost_one_epoch, device
from models.utils.WaveGANUtils import WaveGANUtils
from models.utils.file_logger import file_logger
from models.utils.visualization.visualization import plot_loss

wave_gan_utils = WaveGANUtils()


class WaveGAN:
    def __init__(self):
        self.Logger = file_logger('wavegan')
        self.Logger.start()

        # =============Set Parameters===============
        self.epochs, self.batch_size, self.latent_dim, self.ngpus, self.model_size, self.model_dir, \
        self.epochs_per_sample, self.lmbda, audio_dir, self.output_dir, arguments = get_params()

        # network
        self.netG, self.netD = wave_gan_utils.create_network(self.model_size, self.ngpus, self.latent_dim)

        # "Two time-scale update rule"(TTUR) to update netD 4x faster than netG.
        self.optimizerG, self.optimizerD = wave_gan_utils.optimizers(arguments)

        # Sample noise used for generated output.
        self.sample_noise = wave_gan_utils.sample_noise(arguments, self.latent_dim, device)

        # Save config.
        self.Logger.save_configurations()
        creat_dump(self.model_dir, arguments)

        # Load data.
        self.Logger.loading_data()
        # self.BATCH_NUM, self.train_iter, self.valid_iter, self.test_iter = self.dataset.split_manage_data(arguments,
        #                                                                                                   self.batch_size)

        # collect cost information
        self.D_cost_train, self.D_wass_train = 0, 0

        self.history, self.G_costs = [], []
        self.D_costs_train, self.D_wasses_train = [], []
        self.D_costs_valid, self.D_wasses_valid = [], []

    # def train(self):
    #     # =============Train===============
    #     start = time.time()
    #     self.Logger.start_training(self.epochs, self.batch_size, self.BATCH_NUM)
    #     for epoch in range(1, self.epochs + 1):
    #         # self.Logger.info("{} Epoch: {}/{}".format(time_since(start), epoch, self.epochs))
    #         self.Logger.epoch_info(start, epoch, self.epochs)
    #
    #         D_cost_train_epoch, D_wass_train_epoch = [], []
    #         D_cost_valid_epoch, D_wass_valid_epoch = [], []
    #         G_cost_epoch = []
    #         for i in range(1, self.BATCH_NUM + 1):
    #             #############################
    #             # (1) Train Discriminator (n_discriminate_train times)
    #             #############################
    #             # Set Discriminators parameters to require gradients.
    #             require_net_update(self.netD)
    #
    #             one = torch.tensor(1, dtype=torch.float)
    #             neg_one = torch.tensor(-1, dtype=torch.float)
    #             one = one.to(device)
    #             neg_one = neg_one.to(device)
    #
    #             n_discriminate_train = 5
    #             #############################
    #             # (1.1) Train Discriminator 1 times
    #             #############################
    #             self._train_epoch(D_cost_train_epoch, D_cost_valid_epoch, D_wass_train_epoch, D_wass_valid_epoch,
    #                               n_discriminate_train, neg_one, one)
    #
    #             #############################
    #             # (3) Train Generator
    #             #############################
    #             # Prevent discriminator update.
    #             self.train_generator(neg_one, G_cost_epoch, start, epoch, i)
    #
    #         # Save the average cost of batches in every epoch.
    #         save_avg_cost_one_epoch(D_cost_train_epoch, D_wass_train_epoch, D_cost_valid_epoch, D_wass_valid_epoch,
    #                                 G_cost_epoch,
    #                                 self.D_costs_train, self.D_wasses_train, self.D_costs_valid,
    #                                 self.D_wasses_valid, self.G_costs, self.Logger, start)
    #
    #         # Generate audio samples.
    #         if epoch % self.epochs_per_sample == 0:
    #             wave_gan_utils.generate_audio_samples(self.Logger, self.sample_noise, epoch, self.output_dir)
    #
    #             # TODO: Early stopping by Inception Score(IS)
    #
    #     self.Logger.training_finished()
    #
    #     # TODO: Implement check point and load from checkpoint
    #     # Save model
    #     self.Logger.save_model()
    #
    #     self.last_touch()
    #
    #     self.Logger.end()
    #
    # def _train_epoch(self, D_cost_train_epoch, D_cost_valid_epoch, D_wass_train_epoch, D_wass_valid_epoch,
    #                  n_discriminate_train, neg_one, one):
    #     for _ in range(n_discriminate_train):  # train discriminator more than generator by (default)5
    #         noise = self.train_discriminator_once(neg_one, one)
    #
    #         #############################
    #         # (2) Compute Valid data
    #         #############################
    #
    #         self.compute_valid_data(noise, D_cost_train_epoch,
    #                                 D_wass_train_epoch, D_cost_valid_epoch, D_wass_valid_epoch)
    #
    # def train_discriminator_once(self, neg_one, one):
    #     self.netD.zero_grad()
    #     # Noise
    #     noise = torch.Tensor(self.batch_size, self.latent_dim).uniform_(-1, 1)
    #     noise = noise.to(device)
    #     noise.requires_grad = False  # noise_Var = Variable(noise, requires_grad=False)
    #     real_data_Var = numpy_to_var(next(self.train_iter)['X'])
    #     # a) compute loss contribution from real training data
    #     D_real = self.netD(real_data_Var)
    #     D_real = D_real.mean()  # avg loss
    #     D_real.backward(neg_one)  # loss * -1
    #     # b) compute loss contribution from generated data, then backprop.
    #     fake = autograd.Variable(self.netG(noise).data)  # noise_Var
    #     D_fake = self.netD(fake)
    #     D_fake = D_fake.mean()
    #     D_fake.backward(one)
    #     # c) compute gradient penalty and backprop
    #     gradient_penalty = calc_gradient_penalty(self.netD, real_data_Var.data,
    #                                              fake.data, self.batch_size, self.lmbda,
    #                                              device=device)
    #     gradient_penalty.backward(one)
    #     # Compute cost * Wassertein loss..
    #     self.D_cost_train, self.D_wass_train = wassertein_loss(D_fake, D_real, gradient_penalty)
    #     # Update gradient of discriminator.
    #     self.optimizerD.step()
    #     return noise
    #
    # def compute_valid_data(self, noise, D_cost_train_epoch, D_wass_train_epoch, D_cost_valid_epoch, D_wass_valid_epoch):
    #     self.netD.zero_grad()
    #
    #     valid_data_Var = numpy_to_var(next(self.valid_iter)['X'])
    #     D_real_valid = self.netD(valid_data_Var)
    #     D_real_valid = D_real_valid.mean()  # avg loss
    #
    #     # b) compute loss contribution from generated data, then backprop.
    #     fake_valid = self.netG(noise)  # noise_Var
    #     D_fake_valid = self.netD(fake_valid)
    #     D_fake_valid = D_fake_valid.mean()
    #
    #     # c) compute gradient penalty and backprop
    #     gradient_penalty_valid = calc_gradient_penalty(self.netD, valid_data_Var.data,
    #                                                    fake_valid.data, self.batch_size, self.lmbda,
    #                                                    device=device)
    #
    #     compute_and_record_batch_history(D_fake_valid, D_real_valid, self.D_cost_train, self.D_wass_train,
    #                                      gradient_penalty_valid,
    #                                      D_cost_train_epoch, D_wass_train_epoch, D_cost_valid_epoch,
    #                                      D_wass_valid_epoch)
    #
    # def train_generator(self, neg_one, G_cost_epoch, start, epoch, i):
    #     prevent_net_update(self.netD)
    #
    #     # Reset generator gradients
    #     self.netG.zero_grad()
    #
    #     # Noise
    #     noise = torch.tensor(self.batch_size, self.latent_dim).uniform_(-1, 1)
    #     noise = noise.to(device)
    #     noise.requires_grad = False  # noise_Var = Variable(noise, requires_grad=False)
    #
    #     fake = self.netG(noise)  # noise_Var
    #     G = self.netD(fake)
    #     G = G.mean()
    #
    #     # Update gradients.
    #     G.backward(neg_one)
    #     G_cost = -G
    #
    #     self.optimizerG.step()
    #
    #     # Record costs
    #     if cuda:
    #         G_cost = G_cost.cpu()
    #     G_cost_epoch.append(G_cost.data.numpy())
    #
    #     if i % (self.BATCH_NUM // 5) == 0:
    #         self.Logger.batch_info(start, epoch, i, self.BATCH_NUM, self.D_cost_train, self.D_wass_train, G_cost)
    #
    # def last_touch(self) -> None:
    #     """
    #     Last process to end up training. Do it what do you want. Visualization, early stopping, checkpoint of models,
    #     calculate metrics, ...
    #     :return: None
    #     """
    #     utls_basic.save_models(self.output_dir, self.netD, self.netG)
    #     self.Logger.save_loss_curve()
    #     # Plot loss curve.
    #     plot_loss(self.D_costs_train, self.D_wasses_train,
    #               self.D_costs_valid, self.D_wasses_valid, self.G_costs, self.output_dir)


"""
from torch.autograd import Variable
x = Variable(torch.randn(10, 100))
G = WaveGANGenerator(verbose=True, upsample=False)
out = G(x)
print(out.shape)
D = WaveGANDiscriminator(verbose=True)
out2 = D(out)
print(out2.shape)
"""
