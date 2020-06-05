# from models.wavegan import WaveGAN

# 3rd party libraries
import matplotlib.pyplot as plt

# pytorch

# mymodules
from not_functional.models.DataLoader import AudioDataset
from not_functional.models.Trainers.DefaultTrainer import audio_dir, output_dir

import torch
from torch.utils.data import DataLoader
import torchaudio
import torchvision

dataset = AudioDataset(input_dir=audio_dir, output_dir=output_dir)
dataloader = torch.utils.data.DataLoader(dataset, 10, False)
waveforms, labels = next(iter(dataloader))
specgrams = torchaudio.transforms.Spectrogram()(waveforms)
grid = torchvision.utils.make_grid(specgrams)
grid = grid.permute((1, 2, 0)).numpy()

plt.imshow(grid)
plt.show()

print(f"audios: {waveforms}")
print(f"labels: {labels}")
