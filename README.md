# This repository is not a full package yet.

# My Master Thesis
# GAN LAB - version 0.12
Not: Denoiser is current in Denoiser folder. WGAN is not giving good results for denoising.
PyTorch implementation of [!TODO: link]()

Before running, make sure you have the `sc09` dataset, and put that dataset into 
config.py DATASET_NAME variable with full path and filename

If you want to output different directory, change config.py OUTPUT_PATH variable with full path and filename

## Quick Start:
# 1.Installation
```
sudo apt-get install libav-tools
```
<b>I reccomend to look at requirements.txt</b>
### Requirements

```
pip install -r requirements.txt
```

<br/>

# 2. Download dataset
WaveGAN can now be trained on datasets of arbitrary audio files (previously required preprocessing). You can use any folder containing audio, but here are a few example datasets to help you get started:

- [Speech Commands Zero through Nine (SC09)](http://deepyeti.ucsd.edu/cdonahue/wavegan/data/sc09.tar.gz)
- [Drum sound effects](http://deepyeti.ucsd.edu/cdonahue/wavegan/data/drums.tar.gz)
- [Bach piano performances](http://deepyeti.ucsd.edu/cdonahue/wavegan/data/mancini_piano.tar.gz)

### WaveGan Parameters (params.py)
- target_signals_dir: folder including train subfolder contianing train wav data files
- model_prefix: model name used for saving mode
- n_iterations: number of train iterations
- lr_g: generator learning rate
- lr_d: discriminator learning rate
- beta11: Adam optimizer first  decay rate for moment estimates
- beta2:  Adam optimizer second  decay rate for moment estimates
- decay_lr: flag used to decay learning rate linearly through iterations till reaching zero at 100k iteration
- generator_batch_size_factor: in some cases we might try to multiply batch size by a factor when updatng the generator to give it a more correct and meaningful signal from the discriminator
- n_critic: updating the generator every n updates to the critic/ discriminator
- p_coeff: gradient penalty regularization factor
- batch_size: batch size during training default 10
- noise_latent_dim: dimension of the latent dim used to generate waves
- model_capacity_size: capacity of the model default 64 can be 32 when generating longer window length of 2-4 seconds
- output_dir: directory that contains saved model and saved samples during the training
- window_length: window length of the output utterance can be 16384 (1 sec), 32768 (2 sec), 65536 (4 sec)
- manual_seed: model random seed 
- num_channels: to define number of channels used in the data 

# 3.Run

For `sc09` task, **make sure `sc09` dataset under your current project filepath befor run your code.**
```
$ python train.py
```
### Tensorboard Visualization
Run in different console to open Tensorboard.
tensorboard --logdir=runs

##### !!! WARNING !!
If you want to use Tensorboard, you may see some "TracerWarning" warning. This warning throws due to 
MasterThesis/models/custom_transform/custom_transform.py/PhaseShuffle/forward Converting a tensor to a Python index

There is a advice in main documentation
PYTORCH_JIT=0 python main.py
https://pytorch.org/docs/stable/jit.html#disable-jit-for-debugging

#### Training time
* For `SC09` dataset, <don't know yet> takes nearly <don't know yet> to get reasonable result.
* For `piano` piano dataset, <don't know yet> takes <don't know yet> to get reasonable result.
* Increase the `BATCH_SIZE` from 10 to 32 or 64 can acquire shorter per-epoch time on multiple-GPU but slower gradient descent learning rate.

## Results
Generated  ! TODO: soundcloud

Generated  ! TODO: soundcloud

## Contributions
! TODO