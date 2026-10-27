from monai.data import Dataset, DataLoader
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    Resized, RandFlipd, RandRotated, ToTensord, Compose, RepeatChanneld, Lambdad
)
import pandas as pd
import os
import glob
import numpy as np
import torch

LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
    'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia',
]

def build_data_list(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    items = []
    for _, row in df.iterrows():
        findings = row['Finding Labels'].split('|')
        target = [1.0 if lbl in findings else 0.0 for lbl in LABELS]
        # Find the image file in the subdirectories
        image_index = row['Image Index']
        # Search for the file in all subdirectories
        search_pattern = f"{image_dir}/images_*/images/{image_index}"
        matches = glob.glob(search_pattern)
        if matches:
            image_path = matches[0]
        else:
            print(f"Warning: Could not find {image_index}")
            continue
        items.append({'image': image_path, 'label': target})
    return items

def normalize_channels(x):
    """Ensure image has exactly 1 channel by taking mean if multiple channels"""
    if len(x.shape) == 3 and x.shape[0] > 1:
        # If multiple channels, take mean to convert to grayscale
        x = np.mean(x, axis=0, keepdims=True)
    return x

train_transforms = Compose([
    LoadImaged(keys=['image'], image_only=True),
    EnsureChannelFirstd(keys=['image']),
    ScaleIntensityd(keys=['image']),
    Resized(keys=['image'], spatial_size=(224, 224)),
    Lambdad(keys=['image'], func=normalize_channels),
    RepeatChanneld(keys=['image'], repeats=3),
    RandFlipd(keys=['image'], prob=0.5, spatial_axis=0),
    RandRotated(keys=['image'], range_x=0.1, prob=0.3),
    ToTensord(keys=['image']),
])

def collate_fn(batch):
    """Custom collate function to handle labels properly"""
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32)
    return {'image': images, 'label': labels}

if __name__ == '__main__':
    items = build_data_list(
        '/workspace/data/Data_Entry_2017.csv',
        '/workspace/data',
    )
    print(f'Loaded {len(items)} items')
    sample = train_transforms(items[0])
    print(f'Sample shape: {sample["image"].shape}, range: [{sample["image"].min():.3f}, {sample["image"].max():.3f}]')
