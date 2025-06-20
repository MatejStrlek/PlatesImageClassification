import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from license_plate_dataset import LicensePlateDataset
from transforms import get_train_transforms, get_valid_transforms

def create_dataloaders(csv_path, image_size, batch_size):
    df = pd.read_csv(csv_path)
    df['filepaths'] = df['filepaths'].apply(lambda p: os.path.join('plates_dataset', p))

    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['labels'])
    train_df = df[df["data set"] == "train"]
    valid_df = df[df["data set"] == "valid"]

    train_dataset = LicensePlateDataset(train_df, image_size=image_size, transform=get_train_transforms(image_size))
    valid_dataset = LicensePlateDataset(valid_df, image_size=image_size, transform=get_valid_transforms(image_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    label_map = dict(enumerate(label_encoder.classes_))
    return train_loader, valid_loader, label_map