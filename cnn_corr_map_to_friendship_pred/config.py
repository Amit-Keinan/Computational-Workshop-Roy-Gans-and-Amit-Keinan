import torch

EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_CLASSES = 4  # e.g., friendship levels: 0, 1, 2, 3
INPUT_LENGTH = 90  # Length of the 1D correlation map input
TRAIN_TEST_SPLIT = 0.8
NUM_ROIS = 100  # Number of ROIs, corresponds to input feature length
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SUBJS_CORR_DATA_CSV_PATH = './data/social_data.csv'  # Path to your CSV data file
FRIENDSHIP_LABELS_CSV_PATH = './data/social_distances.csv'  # Path to your labels CSV file