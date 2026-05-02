import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from config import (
    IMG_DIR_1, IMG_DIR_2, CSV_PATH, CLASS_NAMES,
    resize_x, resize_y, imagenet_mean, imagenet_std,
    batch_size, random_seed
)

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}


def get_image_path(image_id):
    for folder in [IMG_DIR_1, IMG_DIR_2]:
        path = os.path.join(folder, image_id + '.jpg')
        if os.path.exists(path):
            return path
    return None


def build_dataframes():
    df = pd.read_csv(CSV_PATH)
    df['filepath'] = df['image_id'].apply(get_image_path)
    df = df.dropna(subset=['filepath'])
    df['label'] = df['dx'].map(CLASS_TO_IDX)

    unique_lesions = df.drop_duplicates(subset=['lesion_id'])[['lesion_id', 'label']]

    train_ids, temp_ids = train_test_split(
        unique_lesions['lesion_id'],
        test_size=0.30,
        stratify=unique_lesions['label'],
        random_state=random_seed
    )
    temp_labels = unique_lesions.set_index('lesion_id').loc[temp_ids, 'label']
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50,
        stratify=temp_labels,
        random_state=random_seed
    )

    train_df = df[df['lesion_id'].isin(train_ids)].reset_index(drop=True)
    val_df   = df[df['lesion_id'].isin(val_ids)].reset_index(drop=True)
    test_df  = df[df['lesion_id'].isin(test_ids)].reset_index(drop=True)

    return train_df, val_df, test_df, df


train_transforms = transforms.Compose([
    transforms.Resize((resize_x + 32, resize_y + 32)),
    transforms.RandomCrop(resize_x),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

eval_transforms = transforms.Compose([
    transforms.Resize((resize_x, resize_y)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])


class SkinLesionDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(row['filepath']).convert('RGB')
        label = int(row['label'])
        if self.transform:
            img = self.transform(img)
        return img, label


def build_dataloaders():
    train_df, val_df, test_df, full_df = build_dataframes()

    train_ds = SkinLesionDataset(train_df, transform=train_transforms)
    val_ds   = SkinLesionDataset(val_df,   transform=eval_transforms)
    test_ds  = SkinLesionDataset(test_df,  transform=eval_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, train_df, test_df


the_dataloader = build_dataloaders
