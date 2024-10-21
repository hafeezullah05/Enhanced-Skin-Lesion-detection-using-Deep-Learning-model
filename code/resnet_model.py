# resnet_model.py
import torch
import torch.nn as nn
import torchvision.models as models

class ResNetModel(nn.Module):
    def __init__(self, num_metadata_features):
        super(ResNetModel, self).__init__()
        
        # Load a pre-trained ResNet-18 model
        self.resnet = models.resnet18(pretrained=True)
        
        # Modify the fully connected layer to accommodate the output of ResNet
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the original classification layer
        
        # Define additional layers for metadata processing
        self.metadata_fc = nn.Sequential(
            nn.Linear(num_metadata_features, 64),  # Example hidden layer
            nn.ReLU(),
            nn.Linear(64, 32),  # Another hidden layer
            nn.ReLU()
        )
        
        # Combine features from ResNet and metadata
        self.combined_fc = nn.Sequential(
            nn.Linear(in_features + 32, 64),  # Combine ResNet features and metadata features
            nn.ReLU(),
            nn.Linear(64, 1)  # Output layer for binary classification
        )
        
    def forward(self, images, metadata):
        # Pass images through the ResNet model
        resnet_features = self.resnet(images)
        
        # Process metadata
        metadata_features = self.metadata_fc(metadata)
        
        # Concatenate ResNet features and metadata features
        combined_features = torch.cat((resnet_features, metadata_features), dim=1)
        
        # Pass through the final fully connected layers
        output = self.combined_fc(combined_features)
        
        return output
