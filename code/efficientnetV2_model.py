import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms

class EfficientNetV2Model(nn.Module):
    def __init__(self, num_metadata_features):
        super(EfficientNetV2Model, self).__init__()

        # Load a pre-trained EfficientNet V2 model
        self.efficientnet = timm.create_model('tf_efficientnetv2_l', pretrained=True)

        # Modify the classifier to remove the original classification layer
        in_features = self.efficientnet.num_features  # Get the number of features from EfficientNet
        self.efficientnet.classifier = nn.Identity()  # Remove the original classification layer

        # Define additional layers for metadata processing
        self.metadata_fc = nn.Sequential(
            nn.Linear(num_metadata_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Combine features from EfficientNet and metadata
        self.classifier = nn.Sequential(
            nn.Linear(in_features + 32, 128),  # Adjust based on the actual combined size
            nn.ReLU(),
            nn.Linear(128, 1)  # Output layer for binary classification
        )

    def forward(self, images, metadata):
        # Pass images through the EfficientNet model
        efficientnet_features = self.efficientnet(images)

        # Process metadata
        metadata_features = self.metadata_fc(metadata)

        # Concatenate EfficientNet features and metadata features
        combined_features = torch.cat((efficientnet_features, metadata_features), dim=1)

        # Pass through the final fully connected layers
        output = self.classifier(combined_features)

        return output