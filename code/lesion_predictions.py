import os
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
from torchvision import transforms

class LesionPredictions:
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.transform_back = transforms.ToPILImage()

    @staticmethod
    def load_metadata(csv_path):
        return pd.read_csv(csv_path)

    def inference_prediction(self):
        self.model.eval()
        all_images = []
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for images, metadata, targets in self.test_loader:  # Removed 'image_names' from the DataLoader
                images, metadata = images.to(self.device), metadata.to(self.device)
                targets = targets.to(self.device)
                
                # Make predictions
                outputs = self.model(images, metadata)
                preds = torch.sigmoid(outputs).round()  # Convert logits to binary predictions (0 or 1)
                
                # Store images, targets, and predictions
                all_images.extend(images.cpu().detach())
                all_targets.extend(targets.cpu().detach().numpy())
                all_preds.extend(preds.cpu().detach().numpy())
        
        # Randomly select 20 samples
        indices = random.sample(range(len(all_images)), 10)
        
        plt.figure(figsize=(20, 40))
        for i, idx in enumerate(indices):
            original_label = "malignant" if all_targets[idx] == 1 else "benign"
            predicted_label = "malignant" if all_preds[idx] == 1 else "benign"

            # Convert image tensor back to PIL image
            img = self.transform_back(all_images[idx])
            
            # Plot the image and prediction details
            plt.subplot(10, 2, i + 1)  # Corrected the subplot number
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Original: {original_label}\nPredicted: {predicted_label}")
            
        plt.tight_layout()
        plt.show()
