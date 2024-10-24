# preprocessing_csv.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class PreprocessingCSV:
    def __init__(self, csv_path, base_dir):
        self.csv_path = csv_path
        self.base_dir = base_dir
        self.clean_csv_path = os.path.join(base_dir, 'Thesis_Hafeez/Dataset/Train_JPEG/ISIC_2020_Training_GroundTruth_preprocess.csv')
        self.df = pd.read_csv(csv_path)

    def analyze_raw_data(self):
        # Ensure 'anatom_site_general_challenge' is treated as a string and strip spaces
        self.df['anatom_site_general_challenge'] = self.df['anatom_site_general_challenge'].astype(str).str.strip()

        # Print unique values to identify anomalies
        unique_values = self.df['anatom_site_general_challenge'].unique()
        print("\n \n Unique values in 'anatom_site_general_challenge':", unique_values)

        # Check for rows that may contain 'unknown'
        unknown_variations = self.df[self.df['anatom_site_general_challenge'].str.contains('unknown', case=False, na=False)]
        print("\n \n Rows with variations of 'unknown':\n", unknown_variations)

        # Replace common variations with 'unknown'
        self.df['anatom_site_general_challenge'] = self.df['anatom_site_general_challenge'].replace({
            'nan': 'unknown',
            'lower extremityi wrote to him': 'lower extremity',
            # Add other replacements if needed
        })

        # Verify unique values after cleanup
        unique_values = self.df['anatom_site_general_challenge'].unique()
        print("\n \n Unique values in 'anatom_site_general_challenge' after cleanup:", unique_values)

    def check_for_anomalies(self):
        # Check for NaN values
        nan_values = self.df.isna().sum()
        print("\n \n NaN values in each column:\n", nan_values)

        # Check for infinity values
        infinity_values = self.df.replace([np.inf, -np.inf], np.nan).isna().sum() - self.df.isna().sum()
        print("\n\n Infinity values in each column:\n", infinity_values)

        # Check for 'unknown' values
        unknown_values = (self.df['anatom_site_general_challenge'] == 'unknown').sum()
        print("\n \n Number of 'unknown' values in 'anatom_site_general_challenge':", unknown_values)

        # Check for unusually large values
        columns_for_max = [col for col in self.df.columns if col != 'diagnosis']
        max_values = self.df[columns_for_max].max(numeric_only=True)
        print("\n \nMaximum values in each column (excluding 'diagnosis'):\n", max_values)

    def clean_data(self):
        # Clean the DataFrame
        self.df['anatom_site_general_challenge'] = self.df['anatom_site_general_challenge'].astype(str).str.strip()
        self.df.replace('nan', np.nan, inplace=True)
        self.df = self.df[self.df['anatom_site_general_challenge'] != 'unknown']
        self.df = self.df.replace([np.inf, -np.inf], np.nan).dropna()

        # Drop the 'diagnosis' column if it exists
        if 'diagnosis' in self.df.columns:
            self.df = self.df.drop(columns=['diagnosis'])

    def save_clean_data(self):
        # Save the cleaned DataFrame to a new CSV
        self.df.to_csv(self.clean_csv_path, index=False)
        print(f"\n \n Preprocessed CSV saved to {self.clean_csv_path}.")

        # Reload the cleaned data for further processing
        self.df = pd.read_csv(self.clean_csv_path)

    def split_by_patient_id(df, base_dir):
        # Define train and test CSV paths
        train_csv_path = os.path.join(base_dir, 'Thesis_Hafeez/Dataset/split_csv/train_split.csv')
        test_csv_path = os.path.join(base_dir, 'Thesis_Hafeez/Dataset/split_csv/test_split.csv')

       # Check if DataFrame is empty
        if df.empty:
            print("Warning: Input DataFrame is empty. Please provide a valid DataFrame.")
            return
    
         # Check if 'patient_id' column exists
        if 'patient_id' not in df.columns:
            print("Warning: 'patient_id' column is missing from the DataFrame.")
            return

        # Split the dataset by patient_id
        unique_patient_ids = df['patient_id'].unique()

        # Check if there are enough unique patient IDs to split
        if len(unique_patient_ids) < 2:
            print("Warning: Not enough unique patient IDs to split the dataset. At least two unique IDs are required.")
            return

        # Split unique patient ids into train and test sets
        train_patient_ids, test_patient_ids = train_test_split(unique_patient_ids, test_size=0.25, random_state=42)

        # Ensure train and test DataFrames are not empty
        train_df = df[df['patient_id'].isin(train_patient_ids)]
        test_df = df[df['patient_id'].isin(test_patient_ids)]

        if train_df.empty or test_df.empty:
            print("Warning: The resulting train or test DataFrame is empty. Please check the dataset for consistency.")
            return

        # Create or empty the CSV file if it exists
        if os.path.exists(train_csv_path):
            open(train_csv_path, 'w').close()  # Empty the file if it exists
        if os.path.exists(test_csv_path):
            open(test_csv_path, 'w').close()  # Empty the file if it exists

        # Write train and test DataFrames to CSV
        train_df.to_csv(train_csv_path, index=False)
        test_df.to_csv(test_csv_path, index=False)

        print("\nTrain and test CSV files have been written to 'split_csv'.")

    def write_csv(self, dataframe, path):
        # Write the DataFrame to CSV
        dataframe.to_csv(path, index=False)

    def verify_preprocessed_data(self):
        # Re-load the cleaned CSV
        df_cleaned = pd.read_csv(self.clean_csv_path)

        # Check for NaN values
        nan_values = df_cleaned.isna().sum()
        print("\n \n NaN values in each column:\n", nan_values)

        # Check for infinity values
        infinity_values = df_cleaned.replace([np.inf, -np.inf], np.nan).isna().sum() - df_cleaned.isna().sum()
        print("\n \n Infinity values in each column:\n", infinity_values)

        # Check for 'unknown' values
        unknown_values = (df_cleaned['anatom_site_general_challenge'] == 'unknown').sum()
        print("\n \n Number of 'unknown' values in 'anatom_site_general_challenge':", unknown_values)

        # Print unique values
        unique_values = df_cleaned['anatom_site_general_challenge'].unique()
        print("\n \n Unique values in 'anatom_site_general_challenge':", unique_values)

        # Check for unusually large values
        columns_for_max = [col for col in df_cleaned.columns if col != 'diagnosis']
        max_values = df_cleaned[columns_for_max].max(numeric_only=True)
        print("\n \n Maximum values in each column (excluding 'diagnosis'):\n", max_values)
