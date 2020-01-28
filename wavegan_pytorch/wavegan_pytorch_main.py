from wavegan_pytorch.models.Trainers.train import WaveGan_GP
from wavegan_pytorch.utils import *
from wavegan_pytorch.utils import WavDataLoader


def main():
    train_loader: WavDataLoader = WavDataLoader(os.path.join(target_signals_dir, 'train'))
    val_loader: WavDataLoader = WavDataLoader(os.path.join(target_signals_dir, 'valid'))

    wave_gan: WaveGan_GP = WaveGan_GP(train_loader, val_loader)
    wave_gan.train()
    visualize_loss(wave_gan.g_cost, wave_gan.valid_g_cost, 'Train', 'Val', 'Negative Critic Loss')
    latent_space_interpolation(wave_gan.generator, n_samples=5)


if __name__ == '__main__':
    print("main Started")
    main()
    print("main Ended")
