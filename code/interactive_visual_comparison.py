import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
from torchvision import transforms

# Load the test CSV to get metadata information
def load_metadata(csv_path):
    return pd.read_csv(csv_path)

# Define Function for Interactive Visual Comparison of 20 Random Images
def interactive_visual_comparison(model, test_loader, device, test_metadata_df):
    model.eval()
    all_images = []
    all_image_names = []
    all_targets = []
    all_preds = []
    transform_back = transforms.ToPILImage()

    with torch.no_grad():
        for images, metadata, targets in test_loader:
            images, metadata = images.to(device), metadata.to(device)
            targets = targets.to(device)
            
            # Make predictions
            outputs = model(images, metadata)
            preds = torch.sigmoid(outputs).round()  # Convert logits to binary predictions (0 or 1)
            
            # Store images, image names, targets, and predictions
            all_images.extend(images.cpu().detach())
            all_image_names.extend(metadata[:, 0].cpu().detach())  # Assuming image names are in metadata
            all_targets.extend(targets.cpu().detach().numpy())
            all_preds.extend(preds.cpu().detach().numpy())
    
    # Randomly select 20 samples
    indices = random.sample(range(len(all_images)), 20)
    
    plt.figure(figsize=(20, 40))
    for i, idx in enumerate(indices):
        img_name = all_image_names[idx]
        original_label = "malignant" if all_targets[idx] == 1 else "benign"
        predicted_label = "malignant" if all_preds[idx] == 1 else "benign"

        # Fetch metadata from the CSV file
        meta_row = test_metadata_df[test_metadata_df['image_name'] == img_name]
        benign_malignant = meta_row['benign_malignant'].values[0]
        target = int(meta_row['target'].values[0])

        # Convert image tensor back to PIL image
        img = transform_back(all_images[idx])
        
        # Plot the image and metadata
        plt.subplot(10, 2, i * 2 + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Image: {img_name}")

        # Plot metadata and prediction details
        plt.subplot(10, 2, i * 2 + 2)
        plt.axis('off')
        plt.text(0.1, 0.8, f"Original: {benign_malignant} (Target: {target})", fontsize=12)
        plt.text(0.1, 0.6, f"Predicted: {predicted_label} (Predicted Target: {int(all_preds[idx])})", fontsize=12)
    
    plt.tight_layout()
    plt.show()