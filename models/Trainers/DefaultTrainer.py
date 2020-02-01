# This trainer faster but not good as WaveganTrainer.py
# inspired from https://github.com/EmilienDupont/wgan-gp/blob/master/training.py
# Standart library
import time
from typing import NoReturn

# torch modules
import torch
from torch import autograd

import models.utils.BasicUtils
from models.Trainers.TrainingUtility import TrainingUtility
# from models.GANSelector import epochs, batch_size, latent_dim, epochs_per_sample, lmbda, output_dir, arguments
# my modules
from models.architectures.WaveGAN import wave_gan_utils
from models.losses.BaseLoss import wassertein_loss
from models.utils.BasicUtils import get_params
from models.utils.BasicUtils import require_net_update, numpy_to_var, calc_gradient_penalty, \
    prevent_net_update, cuda, compute_and_record_batch_history, save_avg_cost_one_epoch
from config import device
# 3rd party modules


torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)  # On by default, leave it here for clarity

# =============Set Parameters===============
epochs, batch_size, latent_dim, ngpus, model_size, model_dir, \
epochs_per_sample, lmbda, audio_dir, output_dir, arguments = get_params()


# TODO: Change and Complete. Tranfer WaveGAN_standalone.py train functions to here
class DefaultTrainer:
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer, GAN, data_loader=None,
                 gp_weight=10, critic_iterations=5, print_every=50,
                 use_cuda=True):
        # cuda = True if torch.cuda.is_available() and use_cuda else False
        # self.device = torch.device("cuda" if cuda else "cpu")
        print("Training device: {}".format(device))

        self.data_loader = data_loader
        self.BATCH_NUM, self.train_iter, self.valid_iter, self.test_iter = \
            data_loader.split_manage_data(arguments, batch_size)

        self.GAN = GAN
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}
        self.num_steps = 0
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every

        # Sample noise used for generated output.
        self.sample_noise = models.utils.BasicUtils.sample_noise(arguments, latent_dim, device)

        # collect cost information
        self.D_cost_train, self.D_wass_train = 0, 0

        self.history, self.G_costs = [], []
        self.D_costs_train, self.D_wasses_train = [], []
        self.D_costs_valid, self.D_wasses_valid = [], []

    def train(self):
        # =============Train===============
        start = time.time()
        self.GAN.Logger.start_training(epochs, batch_size, self.BATCH_NUM)
        self.train_all_epochs(start)

        self.GAN.Logger.training_finished()

        # TODO: Implement check point and load from checkpoint
        # Save model
        self.GAN.Logger.save_model()

        self.last_touch()

        self.GAN.Logger.end()

    def train_all_epochs(self, start):
        for epoch in range(1, epochs + 1):
            self.train_one_epoch(epoch, start)

    def train_one_epoch(self, epoch, start):
        # self.GAN.Logger.info("{} Epoch: {}/{}".format(time_since(start), epoch, self.epochs))
        self.GAN.Logger.epoch_info(start, epoch, epochs)
        D_cost_train_epoch, D_wass_train_epoch = [], []
        D_cost_valid_epoch, D_wass_valid_epoch = [], []
        G_cost_epoch = []
        self.train_gan_all_batches(D_cost_train_epoch, D_cost_valid_epoch, D_wass_train_epoch, D_wass_valid_epoch,
                                   G_cost_epoch, start, epoch)
        # Save the average cost of batches in every epoch.
        save_avg_cost_one_epoch(D_cost_train_epoch, D_wass_train_epoch, D_cost_valid_epoch, D_wass_valid_epoch,
                                G_cost_epoch,
                                self.D_costs_train, self.D_wasses_train, self.D_costs_valid,
                                self.D_wasses_valid, self.G_costs, self.GAN.Logger, start)
        # Generate audio samples.
        if epoch % epochs_per_sample == 0:
            wave_gan_utils.generate_audio_samples(self.GAN.Logger, self.sample_noise, epoch, output_dir)

            # TODO: Early stopping by Inception Score(IS)

    def train_gan_all_batches(self, D_cost_train_epoch, D_cost_valid_epoch, D_wass_train_epoch, D_wass_valid_epoch,
                              G_cost_epoch, start, epoch):
        for i in range(1, self.BATCH_NUM + 1):
            self.train_gan_one_batch(D_cost_train_epoch, D_cost_valid_epoch, D_wass_train_epoch, D_wass_valid_epoch,
                                     G_cost_epoch, start, epoch, i)

    def train_gan_one_batch(self, D_cost_train_epoch, D_cost_valid_epoch, D_wass_train_epoch, D_wass_valid_epoch,
                            G_cost_epoch, start, epoch, i):
        #############################
        # (1) Train Discriminator (n_discriminate_train times)
        #############################
        # Set Discriminators parameters to require gradients.
        require_net_update(self.D)

        one = torch.tensor(1, dtype=torch.float)
        neg_one = torch.tensor(-1, dtype=torch.float)
        one = one.to(device)
        neg_one = neg_one.to(device)

        #############################
        # (1.1) Train Discriminator 1 times
        #############################
        self.train_discriminator(D_cost_train_epoch, D_cost_valid_epoch, D_wass_train_epoch, D_wass_valid_epoch,
                                 self.critic_iterations, neg_one, one)

        #############################
        # (3) Train Generator
        #############################
        # Prevent discriminator update.
        self.train_generator_once(neg_one, G_cost_epoch, start, epoch, i)

    def train_discriminator(self, D_cost_train_epoch, D_cost_valid_epoch, D_wass_train_epoch, D_wass_valid_epoch,
                            n_discriminate_train, neg_one, one):
        for _ in range(n_discriminate_train):  # train discriminator more than generator by (default)5
            noise = self.train_discriminator_once(neg_one, one)

            #############################
            # (2) Compute Valid data
            #############################

            self.compute_valid_data(noise, D_cost_train_epoch,
                                    D_wass_train_epoch, D_cost_valid_epoch, D_wass_valid_epoch)

    def train_discriminator_once(self, neg_one, one):
        self.D.zero_grad()
        # Noise
        noise = torch.Tensor(batch_size, latent_dim).uniform_(-1, 1)
        noise = noise.to(device)
        noise.requires_grad = False  # noise_Var = Variable(noise, requires_grad=False)
        real_data_Var = numpy_to_var(next(self.train_iter)['X'])
        # a) compute loss contribution from real training data
        D_real = self.D(real_data_Var)
        D_real = D_real.mean()  # avg loss
        D_real.backward(neg_one)  # loss * -1
        # b) compute loss contribution from generated data, then backprop.
        fake = autograd.Variable(self.G(noise).data)  # noise_Var
        D_fake = self.D(fake)
        D_fake = D_fake.mean()
        D_fake.backward(one)
        # c) compute gradient penalty and backprop
        gradient_penalty = calc_gradient_penalty(self.D, real_data_Var.data,
                                                 fake.data, batch_size, lmbda,
                                                 device=device)
        gradient_penalty.backward(one)
        # Compute cost * Wassertein loss..
        self.D_cost_train, self.D_wass_train = wassertein_loss(D_fake, D_real, gradient_penalty)
        # Update gradient of discriminator.
        self.D_opt.step()
        return noise

    def compute_valid_data(self, noise, D_cost_train_epoch, D_wass_train_epoch, D_cost_valid_epoch, D_wass_valid_epoch):
        self.D.zero_grad()

        valid_data_Var = numpy_to_var(next(self.valid_iter)['X'])
        D_real_valid = self.D(valid_data_Var)
        D_real_valid = D_real_valid.mean()  # avg loss

        # b) compute loss contribution from generated data, then backprop.
        fake_valid = self.G(noise)  # noise_Var
        D_fake_valid = self.D(fake_valid)
        D_fake_valid = D_fake_valid.mean()

        # c) compute gradient penalty and backprop
        gradient_penalty_valid = calc_gradient_penalty(self.D, valid_data_Var.data,
                                                       fake_valid.data, batch_size, lmbda,
                                                       device=device)

        compute_and_record_batch_history(D_fake_valid, D_real_valid, self.D_cost_train, self.D_wass_train,
                                         gradient_penalty_valid,
                                         D_cost_train_epoch, D_wass_train_epoch, D_cost_valid_epoch,
                                         D_wass_valid_epoch)

    def train_generator_once(self, neg_one, G_cost_epoch, start, epoch, i):
        prevent_net_update(self.D)

        # Reset generator gradients
        self.G.zero_grad()

        # Noise
        noise = torch.Tensor(batch_size, latent_dim).uniform_(-1, 1)
        noise = noise.to(device)
        noise.requires_grad = False  # noise_Var = Variable(noise, requires_grad=False)

        fake = self.G(noise)  # noise_Var
        G = self.D(fake)
        G = G.mean()

        # Update gradients.
        G.backward(neg_one)
        G_cost = -G

        self.G_opt.step()

        # Record costs
        if cuda:
            G_cost = G_cost.cpu()
        G_cost_epoch.append(G_cost.data.numpy())

        if i % (self.BATCH_NUM // 5) == 0:
            self.GAN.Logger.batch_info(start, epoch, i, self.BATCH_NUM, self.D_cost_train, self.D_wass_train, G_cost)

    def last_touch(self) -> NoReturn:
        TrainingUtility.last_touch(self.GAN, self.D, self.G, output_dir, self.D_costs_train, self.D_wasses_train,
                                   self.D_costs_valid, self.D_wasses_valid, self.G_costs)

    @property
    def manager(self):
        return self.manager

    @manager.setter
    def manager(self, m):
        self.manager = m

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Don't REMOVE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # def _critic_train_iteration(self, data):
    #     """ """
    #     # Get generated data
    #     batch_size = data.size()[0]
    #     generated_data = self.sample_generator(batch_size)
    #
    #     # Calculate probabilities on real and generated data
    #     data.to(self.device)
    #     d_real = self.D(data)
    #     d_generated = self.D(generated_data)
    #
    #     # Get gradient penalty
    #     gradient_penalty = self._gradient_penalty(data, generated_data)
    #     self.losses['GP'].append(gradient_penalty.data[0])
    #
    #     # Create total loss and optimize
    #     self.D_opt.zero_grad()
    #     d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
    #     d_loss.backward()
    #
    #     self.D_opt.step()
    #
    #     # Record loss
    #     self.losses['D'].append(d_loss.data[0])
    #
    # def _generator_train_iteration(self, data):
    #     """ """
    #     self.G_opt.zero_grad()
    #
    #     # Get generated data
    #     batch_size = data.size()[0]
    #     generated_data = self.sample_generator(batch_size)
    #
    #     # Calculate loss and optimize
    #     d_generated = self.D(generated_data)
    #     g_loss = - d_generated.mean()
    #     g_loss.backward()
    #     self.G_opt.step()
    #
    #     # Record loss
    #     self.losses['G'].append(g_loss.data.item())
    #
    # def _gradient_penalty(self, real_data, generated_data):
    #     batch_size = real_data.size()[0]
    #
    #     # Calculate interpolation
    #     alpha = torch.rand(batch_size, 1, 1, 1)
    #     alpha = alpha.expand_as(real_data)
    #     alpha = alpha.to(self.device)
    #     interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    #     interpolated.requires_grad = True
    #     interpolated = interpolated.to(self.device)
    #
    #     # Calculate probability of interpolated examples
    #     prob_interpolated = self.D(interpolated)
    #
    #     # Calculate gradients of probabilities with respect to examples
    #     gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
    #                            grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
    #                            create_graph=True, retain_graph=True)[0]
    #
    #     # Gradients have shape (batch_size, num_channels, img_width, img_height),
    #     # so flatten to easily take norm per example in batch
    #     gradients = gradients.view(batch_size, -1)
    #     self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data.item())  # losses memorizes python numbers
    #
    #     # Derivatives of the gradient close to 0 can cause problems because of
    #     # the square root, so manually calculate norm and add epsilon
    #     gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    #
    #     # Return gradient penalty
    #     return self.gp_weight * ((gradients_norm - 1) ** 2).mean()
    #
    # def train_discriminator(self, data_loader):
    # # def train_discriminator(self, data_loader):
    #     # for i, data in enumerate(data_loader):
    #     for i, data in enumerate(self.train_iter):
    #         self.num_steps += 1
    #         self._critic_train_iteration(data[0])
    #         # Only update generator every |critic_iterations| iterations
    #         if self.num_steps % self.critic_iterations == 0:
    #             self._generator_train_iteration(data[0])
    #
    #         if i % self.print_every == 0:
    #             print("Iteration {}".format(i + 1))
    #             print("D: {}".format(self.losses['D'][-1]))
    #             print("GP: {}".format(self.losses['GP'][-1]))
    #             print("Gradient norm: {}".format(self.losses['gradient_norm'][-1]))
    #             if self.num_steps > self.critic_iterations:
    #                 print("G: {}".format(self.losses['G'][-1]))
    #
    # def train(self, epochs, save_training_gif=True):
    # # def train(self, data_loader, epochs, save_training_gif=True):
    #     if save_training_gif:
    #         # Fix latents to see how image generation improves during training
    #         fixed_latents = self.G.sample_latent(64)
    #         fixed_latents = fixed_latents.to(self.device)
    #         training_progress_images = []
    #
    #     for epoch in range(epochs):
    #         print("\nEpoch {}".format(epoch + 1))
    #         # self.train_discriminator(data_loader)
    #         self.train_discriminator()
    #
    #         if save_training_gif:
    #             # Generate batch of images and convert to grid
    #             img_grid = make_grid(self.G(fixed_latents).cpu().data)
    #             # Convert to numpy and transpose axes to fit imageio convention
    #             # i.e. (width, height, channels)
    #             img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
    #             # Add image grid to training progress
    #             training_progress_images.append(img_grid)
    #
    #     if save_training_gif:
    #         imageio.mimsave('./training_{}_epochs.gif'.format(epochs),
    #                         training_progress_images)
    #
    # def sample_generator(self, num_samples):
    #     latent_samples = self.G.sample_latent(num_samples)
    #     latent_samples = latent_samples.to(self.device)
    #     return self.G(latent_samples)
    #
    # def sample(self, num_samples):
    #     generated_data = self.sample_generator(num_samples)
    #     # Remove color channel
    #     return generated_data.data.cpu().numpy()[:, 0, :, :]


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
