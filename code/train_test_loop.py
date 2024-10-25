import time
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

# Import necessary metric functions from sklearn
from sklearn.metrics import precision_score as skl_precision_score
from sklearn.metrics import recall_score as skl_recall_score
from sklearn.metrics import f1_score as skl_f1_score
from sklearn.metrics import accuracy_score as skl_accuracy_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

# Define metric calculation functions
def accuracy_score(preds, labels):
    preds = torch.sigmoid(preds).round()  # No need for conversion, assume preds is already a tensor
    correct = (preds.squeeze() == labels).sum().item()
    return correct / labels.size(0)

def precision_score(preds, labels):
    preds = torch.sigmoid(preds).round()  # Ensure preds is tensor
    return skl_precision_score(labels.cpu().detach().numpy(), preds.cpu().detach().numpy(), zero_division=0)

def recall_score(preds, labels):
    preds = torch.sigmoid(preds).round()  # Ensure preds is tensor
    return skl_recall_score(labels.cpu().detach().numpy(), preds.cpu().detach().numpy(), zero_division=0)

def f1_score(preds, labels):
    preds = torch.sigmoid(preds).round()  # Ensure preds is tensor
    return skl_f1_score(labels.cpu().detach().numpy(), preds.cpu().detach().numpy(), zero_division=0)

# Training and testing function
def train_and_test(model, train_loader, test_loader, optimizer, scheduler, lossFunc, DEVICE, NUM_EPOCHS):
    H = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
        "train_precision": [],
        "test_precision": [],
        "train_recall": [],
        "test_recall": [],
        "train_f1": [],
        "test_f1": [],
        "train_roc_auc": [],
        "test_roc_auc": [],
        "test_precision_recall_curve": [],
        "test_average_precision": []
    }

    # Calculate weights for each class
    malignant_count = len(train_loader.dataset.metadata[train_loader.dataset.metadata['benign_malignant'] == 'malignant'])
    benign_count = len(train_loader.dataset.metadata[train_loader.dataset.metadata['benign_malignant'] == 'benign'])
    total_count = malignant_count + benign_count
    weight_benign = total_count / (2 * benign_count)
    weight_malignant = total_count / (2 * malignant_count)
    weights = [weight_benign, weight_malignant]

    startTime = time.time()
    
    for epoch in range(NUM_EPOCHS):
        # Print epoch information
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS} running...")
        
        # === Training Loop ===
        model.train()
        running_train_loss = 0.0
        all_train_preds = []
        all_train_targets = []

        with tqdm(total=len(train_loader), desc='Training', unit='batch') as pbar:
            for images, metadata, targets in train_loader:
                images, metadata, targets = images.to(DEVICE), metadata.to(DEVICE), targets.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(images, metadata)
                loss = lossFunc(outputs.squeeze(), targets)
                loss.backward()
                optimizer.step()
                
                running_train_loss += loss.item() * images.size(0)
                
                # Collect predictions and targets
                all_train_preds.extend(outputs.detach())  # Keep as tensors
                all_train_targets.extend(targets.detach()) # Keep as tensors
                
                pbar.update(1)  # Update progress bar
        
        avg_train_loss = running_train_loss / len(train_loader.dataset)
        
        # Convert collected tensors to a stacked tensor (needed for metric functions)
        all_train_preds = torch.stack(all_train_preds)  # Convert list of tensors to tensor
        all_train_targets = torch.stack(all_train_targets)  # Convert list of tensors to tensor
        
        # Metrics calculation for training
        train_acc = accuracy_score(all_train_preds, all_train_targets)
        train_precision = precision_score(all_train_preds, all_train_targets)
        train_recall = recall_score(all_train_preds, all_train_targets)
        train_f1 = f1_score(all_train_preds, all_train_targets)
        train_roc_auc = roc_auc_score(all_train_targets.cpu().detach().numpy(), torch.sigmoid(all_train_preds).cpu().detach().numpy())

        # === Testing Loop ===
        model.eval()
        running_test_loss = 0.0
        all_test_preds = []
        all_test_targets = []
        all_test_probs = []

        with torch.no_grad():
            with tqdm(total=len(test_loader), desc='Testing', unit='batch') as pbar:
                for images, metadata, targets in test_loader:
                    images, metadata, targets = images.to(DEVICE), metadata.to(DEVICE), targets.to(DEVICE)
                    
                    outputs = model(images, metadata)
                    loss = lossFunc(outputs.squeeze(), targets)
                    running_test_loss += loss.item() * images.size(0)
                    
                    preds = torch.sigmoid(outputs).round()  # Convert logits to binary predictions (0 or 1)
                    probs = torch.sigmoid(outputs)  # Probabilities for AUC calculation
                    
                    # Collect predictions and true targets
                    all_test_preds.extend(preds.cpu().detach().numpy())
                    all_test_targets.extend(targets.cpu().detach().numpy())
                    all_test_probs.extend(probs.cpu().detach().numpy())
                    
                    pbar.update(1)  # Update progress bar
        
        avg_test_loss = running_test_loss / len(test_loader.dataset)

        # Convert predictions and targets to NumPy arrays
        all_test_preds = np.array(all_test_preds)
        all_test_targets = np.array(all_test_targets)
        all_test_probs = np.array(all_test_probs)

        # Calculate metrics for test set using class weights
        test_acc = skl_accuracy_score(all_test_targets, all_test_preds)
        test_precision = skl_precision_score(all_test_targets, all_test_preds, zero_division=0, sample_weight=[weights[int(target)] for target in all_test_targets])
        test_recall = skl_recall_score(all_test_targets, all_test_preds, zero_division=0, sample_weight=[weights[int(target)] for target in all_test_targets])
        test_f1 = skl_f1_score(all_test_targets, all_test_preds, zero_division=0, sample_weight=[weights[int(target)] for target in all_test_targets])
        test_roc_auc = roc_auc_score(all_test_targets, all_test_probs)

        # Compute Precision-Recall Curve and Average Precision Score
        precision, recall, _ = precision_recall_curve(all_test_targets, all_test_probs)
        average_precision = average_precision_score(all_test_targets, all_test_probs)

        # Store metrics in history
        H["train_loss"].append(avg_train_loss)
        H["test_loss"].append(avg_test_loss)
        H["train_acc"].append(train_acc)
        H["test_acc"].append(test_acc)
        H["train_precision"].append(train_precision)
        H["test_precision"].append(test_precision)
        H["train_recall"].append(train_recall)
        H["test_recall"].append(test_recall)
        H["train_f1"].append(train_f1)
        H["test_f1"].append(test_f1)
        H["train_roc_auc"].append(train_roc_auc)
        H["test_roc_auc"].append(test_roc_auc)
        H["test_precision_recall_curve"].append((precision, recall))
        H["test_average_precision"].append(average_precision)
        
        # Print the results for this epoch
        print(f"[INFO] EPOCH: {epoch + 1}/{NUM_EPOCHS}")
        print(f"Train loss: {avg_train_loss:.6f}")
        print(f"Train Accuracy: {train_acc:.6f}")
        print(f"Train Precision: {train_precision:.6f}")
        print(f"Train Recall: {train_recall:.6f}")
        print(f"Train F1 Score: {train_f1:.6f}")
        print(f"Train ROC AUC: {train_roc_auc:.6f}")
        print(f"Test loss: {avg_test_loss:.6f}")
        print(f"Test Accuracy: {test_acc:.6f}")
        print(f"Test Precision: {test_precision:.6f}")
        print(f"Test Recall: {test_recall:.6f}")
        print(f"Test F1 Score: {test_f1:.6f}")
        print(f"Test ROC AUC: {test_roc_auc:.6f}")
        print(f"Test Average Precision Score: {average_precision:.6f}")
        print()

        # Step the scheduler after every epoch using average training loss
        scheduler.step(avg_train_loss)

    endTime = time.time()
    print(f"[INFO] Total time taken to train the model: {np.round(endTime - startTime)} seconds")
    
    return H
