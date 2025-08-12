import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL_PATH = os.path.abspath("/home/sslab/24m0786/bert-information-bottleneck/local_cache/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594")
DATASET_PATH = os.path.abspath("/home/sslab/24m0786/bert-information-bottleneck/local_cache/DeepPavlov___clinc150")
NUM_LABELS = 151
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-5
BOTTLENECK_DIMS = [8, 32, 64, 128]
# For Part 2: Weight for the reconstruction loss
RECONSTRUCTION_ALPHA = 0.5 
# For Part 3: Standard deviation of the Gaussian noise
NOISE_LEVEL = 0.1