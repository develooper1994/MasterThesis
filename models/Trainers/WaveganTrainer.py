from typing import List, Any, NoReturn

import torch.optim as optim
from torch.autograd import grad
from torch.optim import Adam
from tqdm import tqdm

from config import *
from models.DataLoader.DataLoader import WavDataLoader
from models.Discriminators.WaveGANDiscriminator import WaveGANDiscriminator
from models.Generators.WaveGANGenerator import WaveGANGenerator
from models.utils.BasicUtils import visualize_loss, latent_space_interpolation, sample_noise, gradients_status, \
    save_samples, update_optimizer_lr


class WaveGan_GP(object):
    g_cost: List[Any]
    train_d_cost: List[Any]
    train_w_distance: List[Any]
    valid_d_cost: List[Any]
    valid_w_distance: List[Any]
    valid_g_cost: List[Any]
    valid_reconstruction: List[Any]
    optimizerG: Adam
    optimizerD: Adam
    discriminator: WaveGANDiscriminator
    generator: WaveGANGenerator

    def __init__(self, train_loader: object, val_loader: object, validate: object = True, use_batchnorm: object = False) -> object:
        """

        :rtype: object
        """
        super(WaveGan_GP, self).__init__()
        from models.DefaultTrainBuilder import DefaultRunManager
        self.g_cost = []
        self.train_d_cost = []
        self.train_w_distance = []
        self.valid_d_cost = []
        self.valid_w_distance = []
        self.valid_g_cost = []
        self.valid_reconstruction = []

        self.generated, self.disc_cost, self.disc_wd = 0, 0, 0

        self.discriminator = WaveGANDiscriminator(slice_len=window_length, model_size=model_capacity_size,
                                                  use_batch_norm=use_batchnorm, num_channels=num_channels).to(device)

        self.generator = WaveGANGenerator(slice_len=window_length, model_size=model_capacity_size,
                                          use_batch_norm=use_batchnorm, num_channels=num_channels).to(device)

        self.optimizerG = optim.Adam(self.generator.parameters(), lr=lr_g,
                                     betas=(beta1, beta2))  # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.validate = validate
        self.n_samples_per_batch = len(train_loader)

        self.m = DefaultRunManager(self.discriminator, self.generator, self.val_loader)

    def calculate_discriminator_loss(self, real, generated):
        disc_out_gen = self.discriminator(generated)
        disc_out_real = self.discriminator(real)

        alpha = torch.FloatTensor(batch_size, 1, 1).uniform_(0, 1).to(device)
        alpha = alpha.expand(batch_size, real.size(1), real.size(2))

        interpolated = (1 - alpha) * real.data + (alpha) * generated.data[:batch_size]
        interpolated.requires_grad = True  # = torch.autograd.Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.discriminator(interpolated)
        grad_inputs = interpolated
        ones = torch.ones(prob_interpolated.size()).to(device)
        gradients = grad(outputs=prob_interpolated, inputs=grad_inputs, grad_outputs=ones,
                         create_graph=True, retain_graph=True)[0]
        # calculate gradient penalty
        grad_penalty = p_coeff * ((gradients.view(gradients.size(0), -1).norm(
            2, dim=1) - 1) ** 2).mean()
        assert not (torch.isnan(grad_penalty))
        assert not (torch.isnan(disc_out_gen.mean()))
        assert not (torch.isnan(disc_out_real.mean()))
        cost_wd = disc_out_gen.mean() - disc_out_real.mean()
        cost = cost_wd + grad_penalty
        return cost, cost_wd

    def apply_zero_grad(self) -> NoReturn:
        self.discriminator.zero_grad()
        self.generator.zero_grad()

    def enable_disc_disable_gen(self) -> NoReturn:
        gradients_status(self.discriminator, True)
        gradients_status(self.generator, False)

    def enable_gen_disable_disc(self) -> NoReturn:
        gradients_status(self.discriminator, False)
        gradients_status(self.generator, True)

    def disable_all(self) -> NoReturn:
        gradients_status(self.discriminator, False)
        gradients_status(self.generator, False)

    #############################
    # Train GAN
    #############################
    def train(self):
        progress_bar = tqdm(total=n_iterations // progress_bar_step_iter_size)
        fixed_noise = sample_noise(batch_size).to(device)  # used to save samples every few epochs

        gan_model_name = 'gan_{}.tar'.format(model_prefix)

        first_iter = self.start_training(fixed_noise, gan_model_name, progress_bar)

        self.train_all_epochs(first_iter, fixed_noise, gan_model_name, progress_bar)

        self.generator.eval()

    def train_all_epochs(self, first_iter, fixed_noise, gan_model_name, progress_bar):
        for iter_indx in range(first_iter, n_iterations):
            self.train_one_epoch(fixed_noise, gan_model_name, iter_indx, progress_bar)

    def train_one_epoch(self, fixed_noise, gan_model_name, iter_indx, progress_bar):
        global disc_cost, disc_wd, generated
        self.generator.train()
        self.enable_disc_disable_gen()
        self.train_critic_all()
        self.validate_func(generated, iter_indx)
        #############################
        # (2) Update G network every n_critic steps
        #############################
        generator_cost = self.generator_update()
        self.finish_iteration(disc_cost, disc_wd, fixed_noise, gan_model_name, generator_cost, iter_indx,
                              progress_bar)

    def generator_update(self):
        global generated
        self.apply_zero_grad()
        self.enable_gen_disable_disc()
        noise = sample_noise(batch_size * generator_batch_size_factor).to(device)
        generated = self.generator(noise)
        discriminator_output_fake = self.discriminator(generated)
        generator_cost = -discriminator_output_fake.mean()
        generator_cost.backward()
        self.optimizerG.step()
        return generator_cost

    def finish_iteration(self, disc_cost, disc_wd, fixed_noise, gan_model_name, generator_cost, iter_indx, progress_bar):
        self.cost_store_every(disc_cost, disc_wd, generator_cost, iter_indx, progress_bar)
        if iter_indx % progress_bar_step_iter_size == 0:
            progress_bar.update()
        # lr decay
        self.learning_rate_decay(iter_indx)
        self.save_samples_every(fixed_noise, iter_indx)
        self.saving_dict(gan_model_name, iter_indx)

    def train_critic_all(self):
        for _ in range(n_critic):
            self.train_critic_once()

    def train_critic_once(self):
        global generated, disc_cost, disc_wd
        real_signal, label = next(self.train_loader)
        # need to add mixed signal and flag
        noise = sample_noise(batch_size * generator_batch_size_factor).to(device).to(device)
        generated = self.generator(noise)
        #############################
        # Calculating discriminator loss and updating discriminator
        #############################
        self.apply_zero_grad()
        disc_cost, disc_wd = self.calculate_discriminator_loss(real_signal.data, generated.data)
        assert not (torch.isnan(disc_cost))
        disc_cost.backward()
        self.optimizerD.step()

    def start_training(self, fixed_noise, gan_model_name, progress_bar):
        first_iter = 0
        if take_backup and os.path.isfile(gan_model_name):
            checkpoint = self.load_dict(gan_model_name)

            first_iter = checkpoint['n_iterations']
            for _ in range(0, first_iter, progress_bar_step_iter_size):
                progress_bar.update()
            self.generator.eval()
            with torch.no_grad():
                fake = self.generator(fixed_noise).detach().cpu().numpy()
            save_samples(fake, first_iter)
        return first_iter

    def load_dict(self, gan_model_name):
        if cuda:
            checkpoint = torch.load(gan_model_name)
        else:
            checkpoint = torch.load(gan_model_name, map_location='cpu')
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.optimizerD.load_state_dict(checkpoint['optimizer_d'])
        self.optimizerG.load_state_dict(checkpoint['optimizer_g'])
        self.train_d_cost = checkpoint['train_d_cost']
        self.train_w_distance = checkpoint['train_w_distance']
        self.valid_d_cost = checkpoint['valid_d_cost']
        self.valid_w_distance = checkpoint['valid_w_distance']
        self.valid_g_cost = checkpoint['valid_g_cost']
        self.g_cost = checkpoint['g_cost']
        return checkpoint

    def validate_func(self, generated, iter_indx):
        if self.validate and iter_indx % store_cost_every == 0:
            self.disable_all()
            val_real, label = next(self.val_loader)

            val_disc_loss, val_disc_wd = self.calculate_discriminator_loss(val_real.data, generated.data)
            self.valid_d_cost.append(val_disc_loss.item())
            self.valid_w_distance.append(val_disc_wd.item() * -1)
            val_discriminator_output = self.discriminator(val_real)
            val_generator_cost = val_discriminator_output.mean()
            self.valid_g_cost.append(val_generator_cost.item())

    def cost_store_every(self, disc_cost, disc_wd, generator_cost, iter_indx, progress_bar):
        if iter_indx % store_cost_every == 0:
            self.g_cost.append(generator_cost.item() * -1)
            self.train_d_cost.append(disc_cost.item())
            self.train_w_distance.append(disc_wd.item() * -1)

            progress_updates = {'Loss_D WD': str(self.train_w_distance[-1]), 'Loss_G': str(self.g_cost[-1]),
                                'Val_G': str(self.valid_g_cost[-1])}
            progress_bar.set_postfix(progress_updates)

    def learning_rate_decay(self, iter_indx):
        if decay_lr:
            decay = max(0.0, 1.0 - iter_indx * 1.0 / n_iterations)
            # update the learning rate
            update_optimizer_lr(self.optimizerD, lr_d, decay)
            update_optimizer_lr(self.optimizerG, lr_g, decay)

    def save_samples_every(self, fixed_noise, iter_indx):
        if iter_indx % save_samples_every == 0:
            with torch.no_grad():
                fake = self.generator(fixed_noise).detach().cpu().numpy()
            save_samples(fake, iter_indx)

    def saving_dict(self, gan_model_name, iter_indx):
        if take_backup and iter_indx % backup_every_n_iters == 0:
            saving_dict = {
                'generator': self.generator.state_dict(),
                'discriminator': self.discriminator.state_dict(),
                'n_iterations': iter_indx,
                'optimizer_d': self.optimizerD.state_dict(),
                'optimizer_g': self.optimizerG.state_dict(),
                'train_d_cost': self.train_d_cost,
                'train_w_distance': self.train_w_distance,
                'valid_d_cost': self.valid_d_cost,
                'valid_w_distance': self.valid_w_distance,
                'valid_g_cost': self.valid_g_cost,
                'g_cost': self.g_cost
            }
            torch.save(saving_dict, gan_model_name)

if __name__ == '__main__':
    print("Training Started")
    train_loader: WavDataLoader = WavDataLoader(os.path.join(target_signals_dir, 'train'))
    val_loader: WavDataLoader = WavDataLoader(os.path.join(target_signals_dir, 'valid'))

    wave_gan: WaveGan_GP = WaveGan_GP(train_loader, val_loader)
    wave_gan.train()
    visualize_loss(wave_gan.g_cost, wave_gan.valid_g_cost, 'Train', 'Val', 'Negative Critic Loss')
    latent_space_interpolation(wave_gan.generator, n_samples=5)
    print("Training Ended")
