import os
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
from torchvision import transforms

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

        # Separate malignant and benign samples for dynamic augmentation
        if not is_test:
            self.malignant_samples = self.metadata[self.metadata['benign_malignant'] == 'malignant']
            self.benign_samples = self.metadata[self.metadata['benign_malignant'] == 'benign']
            
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
        benign_malignant = self.metadata.iloc[idx, 5]

        # Encode categorical metadata
        sex_encoded = self.sex_encoder.transform([sex])[0]
        site_encoded = self.site_encoder.transform([site])[0]

        metadata = {
            'sex': sex_encoded,
            'age': age,
            'site': site_encoded
        }
        
        # Convert metadata to tensor
        metadata_tensor = torch.tensor([metadata['sex'], metadata['age'], metadata['site']], dtype=torch.float)
        
        # Define target
        target = torch.tensor(1 if benign_malignant == 'malignant' else 0, dtype=torch.float)
        
        # Apply augmentations dynamically for malignant samples during training
        if not self.is_test and benign_malignant == 'malignant':
            augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = augment_transform(image)
        else:
            if self.transform:
                image = self.transform(image)

        return image, metadata_tensor, target
