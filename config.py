# EmTraining
EPOCHS = 1  # 1 # 10 # 180
# NOTE!!!: Bigger BATCH_SIZE; faster training and slower gradient degradation
BATCH_SIZE = 150  # 193 # 190 # 175 # 150 # 100 # 80 # 64 # 10

SAMPLE_EVERY = 1  # Generate audio samples every 1 epoch.
SAMPLE_NUM = 10  # Generate 10 samples every sample generation.

# signal processing configurations.
# NOTE: DON'T CHANGE IF YOU DON'T REALLLLYYYY NEED
WINDOW_LENGHT = 16384
FS = 16000

# Data
DATASET_NAME = "/home/selcuk/.pytorch/sc09/"  # 'sc09/'
OUTPUT_PATH = "output/"

# Model(Network)
MODEL = "wavegan"
