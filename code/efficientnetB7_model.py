# efficientnet_model.py
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class EfficientNetB7Model(nn.Module):
    def __init__(self, num_metadata_features):
        super(EfficientNetB7Model, self).__init__()
        
        # Load a pre-trained EfficientNet B7 model
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b7') #efficientnet-b7
        #self.efficientnet = models.efficientnet_b7(pretrained=True)

        # Modify the fully connected layer to accommodate the output of EfficientNet
        in_features = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Identity()  # Remove the original classification layer
        
        # Define additional layers for metadata processing
        self.metadata_fc = nn.Sequential(
            nn.Linear(num_metadata_features, 64),  # Example hidden layer
            nn.ReLU(),
            nn.Linear(64, 32),  # Another hidden layer
            nn.ReLU()
        )
        
        # Combine features from EfficientNet and metadata
        self.combined_fc = nn.Sequential(
            nn.Linear(in_features + 32, 64),  # Combine EfficientNet features and metadata features
            nn.ReLU(),
            nn.Linear(64, 1)  # Output layer for binary classification
        )
        
    def forward(self, images, metadata):
        # Pass images through the EfficientNet model
        efficientnet_features = self.efficientnet(images)  # Ensure input size is 600x600
        
        # Process metadata
        metadata_features = self.metadata_fc(metadata)
        
        # Concatenate EfficientNet features and metadata features
        combined_features = torch.cat((efficientnet_features, metadata_features), dim=1)
        
        # Pass through the final fully connected layers
        output = self.combined_fc(combined_features)
        
        return output
