import torch

BASE_DIR  = '/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000'
IMG_DIR_1 = BASE_DIR + '/HAM10000_images_part_1'
IMG_DIR_2 = BASE_DIR + '/HAM10000_images_part_2'
CSV_PATH  = BASE_DIR + '/HAM10000_metadata.csv'
CHECKPOINT_PATH = 'checkpoints/final_weights.pth'

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_FULL  = [
    'Actinic Keratosis', 'Basal Cell Carcinoma',
    'Benign Keratosis', 'Dermatofibroma',
    'Melanoma', 'Melanocytic Nevi', 'Vascular Lesions'
]
num_classes = 7

resize_x = 224
resize_y = 224
input_channels = 3

batch_size    = 32
epochs_phase1 = 10
epochs_phase2 = 15
epochs        = epochs_phase1 + epochs_phase2
learning_rate_phase1 = 1e-3
learning_rate_phase2 = 1e-4
weight_decay  = 1e-4
dropout_rate  = 0.5
hidden_units  = 512

val_split   = 0.15
test_split  = 0.15
random_seed = 42

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
