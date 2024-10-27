import os
import pandas as pd
from PIL import Image
from torchvision import transforms



class TrainMalignantAugmentor:
    def __init__(self, train_aug_csv, image_dir, augmentations_per_image=15):
        self.train_aug_csv = train_aug_csv
        self.image_dir = image_dir
        self.augmentations_per_image = augmentations_per_image
        self.data = pd.read_csv(self.train_aug_csv)
        self.original_data_length = len(self.data)
        
        # Define the augmentation transformations
        self.augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor()
        ])

    def augment_image(self, image_path, image_name_prefix):
        """
        Perform augmentations on a given image and return augmented images with new metadata entries.
        """
        try:
            original_image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return []

        augmented_entries = []

        for i in range(1, self.augmentations_per_image + 1):
            augmented_image = self.augment_transform(original_image)
            augmented_image_name = f"{image_name_prefix}_aug_{i}.JPG"
            augmented_image_path = os.path.join(self.image_dir, augmented_image_name)

            # Save augmented image (specify format explicitly as 'JPEG')
            transforms.ToPILImage()(augmented_image).save(augmented_image_path, format='JPEG')

            # Create metadata entry for the augmented image
            augmented_entries.append(augmented_image_name)

        return augmented_entries

    def perform_augmentation(self):
        augmented_metadata = []

        with tqdm(total=self.original_data_length, desc="Performing Image Augmentation", unit="image") as pbar:
            for idx in range(self.original_data_length):
                row = self.data.iloc[idx]
                image_name = row['image_name']
                benign_malignant = row['benign_malignant']

                # Skip already augmented images
                if "_aug_" in image_name:
                    pbar.update(1)
                    continue

                # Only augment malignant cases
                if benign_malignant == 'malignant':
                    image_path = os.path.join(self.image_dir, image_name)

                    # Check if the file exists
                    if not os.path.exists(image_path):
                        print(f"Image not found: {image_path}")
                        pbar.update(1)
                        continue

                    augmented_image_names = self.augment_image(image_path, image_name)

                    # Append metadata for augmented images
                    for aug_image_name in augmented_image_names:
                        augmented_row = row.copy()
                        augmented_row['image_name'] = aug_image_name
                        augmented_metadata.append(augmented_row)

                pbar.update(1)

        # Append augmented metadata to the original DataFrame
        augmented_metadata_df = pd.DataFrame(augmented_metadata)
        self.data = pd.concat([self.data, augmented_metadata_df], ignore_index=True)

        print("Image augmentations for training dataset is complete.")

    def save_updated_metadata(self):
        """Save the updated metadata to the original CSV file."""
        self.data.to_csv(self.train_aug_csv, index=False)
        print(f"Updated metadata saved to {self.train_aug_csv}")
        print("Process finished successfully.")


class TestMalignantAugmentor:
    def __init__(self, test_aug_csv, image_dir, augmentations_per_image=10):
        self.test_aug_csv = test_aug_csv
        self.image_dir = image_dir
        self.augmentations_per_image = augmentations_per_image
        self.data = pd.read_csv(self.test_aug_csv)
        self.original_data_length = len(self.data)
        
        # Define the augmentation transformations
        self.augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor()
        ])

    def augment_image(self, image_path, image_name_prefix):
        """
        Perform augmentations on a given image and return augmented images with new metadata entries.
        """
        try:
            original_image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return []

        augmented_entries = []

        for i in range(1, self.augmentations_per_image + 1):
            augmented_image = self.augment_transform(original_image)
            augmented_image_name = f"{image_name_prefix}_aug_{i}.JPG"
            augmented_image_path = os.path.join(self.image_dir, augmented_image_name)

            # Save augmented image (specify format explicitly as 'JPEG')
            transforms.ToPILImage()(augmented_image).save(augmented_image_path, format='JPEG')

            # Create metadata entry for the augmented image
            augmented_entries.append(augmented_image_name)

        return augmented_entries

    def perform_augmentation(self):
        augmented_metadata = []

        with tqdm(total=self.original_data_length, desc="Performing Image Augmentation (Test)", unit="image") as pbar:
            for idx in range(self.original_data_length):
                row = self.data.iloc[idx]
                image_name = row['image_name']
                benign_malignant = row['benign_malignant']

                # Skip already augmented images
                if "_aug_" in image_name:
                    pbar.update(1)
                    continue

                # Only augment malignant cases
                if benign_malignant == 'malignant':
                    image_path = os.path.join(self.image_dir, image_name)

                    # Check if the file exists
                    if not os.path.exists(image_path):
                        print(f"Image not found: {image_path}")
                        pbar.update(1)
                        continue

                    augmented_image_names = self.augment_image(image_path, image_name)

                    # Append metadata for augmented images
                    for aug_image_name in augmented_image_names:
                        augmented_row = row.copy()
                        augmented_row['image_name'] = aug_image_name
                        augmented_metadata.append(augmented_row)

                pbar.update(1)

        # Append augmented metadata to the original DataFrame
        augmented_metadata_df = pd.DataFrame(augmented_metadata)
        self.data = pd.concat([self.data, augmented_metadata_df], ignore_index=True)

        print("Image augmentations for test dataset is complete.")

    def save_updated_metadata(self):
        """Save the updated metadata to the original CSV file."""
        self.data.to_csv(self.test_aug_csv, index=False)
        print(f"Updated metadata saved to {self.test_aug_csv}")
        print("Process finished successfully.")



if __name__ == "__main__":
    # Define paths
    BASE_DIR = os.path.join(os.path.expanduser('~'), 'Desktop')
    IMAGE_DIR = os.path.join(BASE_DIR, 'Thesis_Hafeez', 'Dataset', 'Train_JPEG', 'JPEG')
    TRAIN_CSV_PATH = os.path.join(BASE_DIR, 'Thesis_Hafeez', 'Dataset', 'split_csv', 'train_split.csv')
    TEST_CSV_PATH = os.path.join(BASE_DIR, 'Thesis_Hafeez', 'Dataset', 'split_csv', 'test_split.csv')

    # Initialize the augmentor for training data
    train_augmentor = TrainMalignantAugmentor(
        csv_path=TRAIN_CSV_PATH,
        image_dir=IMAGE_DIR
    )

    # Perform augmentation and save the updated metadata for training
    train_augmentor.perform_augmentation()
    train_augmentor.save_updated_metadata()

    # Initialize the augmentor for test data
    test_augmentor = TestMalignantAugmentor(
        csv_path=TEST_CSV_PATH,
        image_dir=IMAGE_DIR
    )

    # Perform augmentation and save the updated metadata for testing
    test_augmentor.perform_augmentation()
    test_augmentor.save_updated_metadata()
