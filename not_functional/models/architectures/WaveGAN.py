# Standart library

# torch imports

# my modules
from not_functional.models.utils.BasicUtils import get_params, creat_dump, device
from not_functional.models.utils.WaveGANUtils import WaveGANUtils
from not_functional.models.utils.file_logger import file_logger

wave_gan_utils = WaveGANUtils()


class WaveGAN:
    def __init__(self) -> None:
        """

        :rtype: 
        """
        self.Logger = file_logger('wavegan')
        self.Logger.start()

        # =============Set Parameters===============
        self.epochs, self.batch_size, self.latent_dim, self.ngpus, self.model_size, self.model_dir, \
        self.epochs_per_sample, self.lmbda, audio_dir, self.output_dir, arguments = get_params()

        # network
        self.generator, self.discriminator = wave_gan_utils.create_network(model_size=self.model_size)

        # "Two time-scale update rule"(TTUR) to update discriminator 4x faster than generator.
        self.optimizerD, self.optimizerG = wave_gan_utils.optimizers(arguments, generator=self.generator, discriminator=self.discriminator)

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
    #             require_net_update(self.discriminator)
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
    #             self.train_discriminator(D_cost_train_epoch, D_cost_valid_epoch, D_wass_train_epoch, D_wass_valid_epoch,
    #                               n_discriminate_train, neg_one, one)
    #
    #             #############################
    #             # (3) Train Generator
    #             #############################
    #             # Prevent discriminator update.
    #             self.train_generator_once(neg_one, G_cost_epoch, start, epoch, i)
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
    # def train_discriminator(self, D_cost_train_epoch, D_cost_valid_epoch, D_wass_train_epoch, D_wass_valid_epoch,
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
    #     self.discriminator.zero_grad()
    #     # Noise
    #     noise = torch.Tensor(self.batch_size, self.latent_dim).uniform_(-1, 1)
    #     noise = noise.to(device)
    #     noise.requires_grad = False  # noise_Var = Variable(noise, requires_grad=False)
    #     real_data_Var = numpy_to_var(next(self.train_iter)['X'])
    #     # a) compute loss contribution from real training data
    #     D_real = self.discriminator(real_data_Var)
    #     D_real = D_real.mean()  # avg loss
    #     D_real.backward(neg_one)  # loss * -1
    #     # b) compute loss contribution from generated data, then backprop.
    #     fake = autograd.Variable(self.generator(noise).data)  # noise_Var
    #     D_fake = self.discriminator(fake)
    #     D_fake = D_fake.mean()
    #     D_fake.backward(one)
    #     # c) compute gradient penalty and backprop
    #     gradient_penalty = calc_gradient_penalty(self.discriminator, real_data_Var.data,
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
    #     self.discriminator.zero_grad()
    #
    #     valid_data_Var = numpy_to_var(next(self.valid_iter)['X'])
    #     D_real_valid = self.discriminator(valid_data_Var)
    #     D_real_valid = D_real_valid.mean()  # avg loss
    #
    #     # b) compute loss contribution from generated data, then backprop.
    #     fake_valid = self.generator(noise)  # noise_Var
    #     D_fake_valid = self.discriminator(fake_valid)
    #     D_fake_valid = D_fake_valid.mean()
    #
    #     # c) compute gradient penalty and backprop
    #     gradient_penalty_valid = calc_gradient_penalty(self.discriminator, valid_data_Var.data,
    #                                                    fake_valid.data, self.batch_size, self.lmbda,
    #                                                    device=device)
    #
    #     compute_and_record_batch_history(D_fake_valid, D_real_valid, self.D_cost_train, self.D_wass_train,
    #                                      gradient_penalty_valid,
    #                                      D_cost_train_epoch, D_wass_train_epoch, D_cost_valid_epoch,
    #                                      D_wass_valid_epoch)
    #
    # def train_generator_once(self, neg_one, G_cost_epoch, start, epoch, i):
    #     prevent_net_update(self.discriminator)
    #
    #     # Reset generator gradients
    #     self.generator.zero_grad()
    #
    #     # Noise
    #     noise = torch.tensor(self.batch_size, self.latent_dim).uniform_(-1, 1)
    #     noise = noise.to(device)
    #     noise.requires_grad = False  # noise_Var = Variable(noise, requires_grad=False)
    #
    #     fake = self.generator(noise)  # noise_Var
    #     G = self.discriminator(fake)
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
    #     utls_basic.save_models(self.output_dir, self.discriminator, self.generator)
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
