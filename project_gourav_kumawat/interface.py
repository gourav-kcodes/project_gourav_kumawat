from model import SkinLesionClassifier as TheModel

from train import train_skin_lesion_model as the_trainer

from predict import classify_skin_lesions as the_predictor

from dataset import SkinLesionDataset as TheDataset

from dataset import build_dataloaders as the_dataloader

from config import batch_size as the_batch_size

from config import epochs as total_epochs
