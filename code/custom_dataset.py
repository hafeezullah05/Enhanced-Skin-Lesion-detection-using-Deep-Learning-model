# custom_dataset.py
import os
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset

class CustomMelanomaDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, is_test=False):
        """
        Args:
            csv_file (string): Path to the CSV file with metadata.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
            is_test (bool): If True, dataset is for testing. If False, dataset is for training.
        """
        self.metadata = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test

        # Initialize label encoders for categorical data
        self.sex_encoder = LabelEncoder().fit(self.metadata['sex'])
        self.site_encoder = LabelEncoder().fit(self.metadata['anatom_site_general_challenge'])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = self.metadata.iloc[idx, 0]  # image_name
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # Load metadata
        sex = self.metadata.iloc[idx, 2]
        age = self.metadata.iloc[idx, 3]
        site = self.metadata.iloc[idx, 4]
        
        # Encode categorical metadata
        sex_encoded = self.sex_encoder.transform([sex])[0]
        site_encoded = self.site_encoder.transform([site])[0]

        metadata = {
            'sex': sex_encoded,
            'age': age,
            'site': site_encoded
        }
        
        # Convert metadata to tensor
        metadata_tensor = torch.tensor([metadata['sex'], metadata['age'], 
        metadata['site']], dtype=torch.float)
        if self.is_test:
            benign_malignant = self.metadata.iloc[idx, 5]
            target = torch.tensor(1 if benign_malignant == 'malignant' else 0, dtype=torch.float)
        
        else:
            # For training data, include benign_malignant and target
            benign_malignant = self.metadata.iloc[idx, 5]
            target = torch.tensor(1 if benign_malignant == 'malignant' else 0, dtype=torch.float)

        if self.transform:
            image = self.transform(image)

        return image, metadata_tensor, target
