# # PLS GOTO README.MD FIRST!!

import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_epoch(model, data_loader, optimizer, scheduler, device):
    """
    Train the model for one epoch
    
    Args:
        model: Model to train
        data_loader: DataLoader for training data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to use for training
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for batch in data_loader:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Calculate loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Update scheduler
        scheduler.step()
        
        # Accumulate loss
        total_loss += loss.item()
    
    # Calculate average loss
    avg_loss = total_loss / len(data_loader)
    
    return avg_loss

def evaluate(model, data_loader, device):
    """
    Evaluate the model
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to use for evaluation
        
    Returns:
        tuple: Average loss, predictions and true labels
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate loss
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Get predictions
            _, preds = torch.max(outputs, dim=1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    # Calculate average loss
    avg_loss = total_loss / len(data_loader)
    
    return avg_loss, all_preds, all_labels

def train_model(model, train_loader, val_loader, optimizer, scheduler, device, epochs=4, 
                gradual_unfreeze=None, patience=3):
    """
    Train the model
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to use for training
        epochs (int): Number of training epochs
        gradual_unfreeze (list): List of epochs to unfreeze more layers
        patience (int): Early stopping patience
        
    Returns:
        tuple: Training history and best model state
    """
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    # Initialize early stopping variables
    best_val_loss = float('inf')
    best_model_state = None
    no_improvement = 0
    
    # Train for specified number of epochs
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Gradual unfreezing
        if gradual_unfreeze is not None and epoch in gradual_unfreeze:
            layers_to_unfreeze = gradual_unfreeze[epoch]
            print(f"Unfreezing {layers_to_unfreeze} layers...")
            model.gradual_unfreeze(layers_to_unfreeze)
        
        # Time the epoch
        start_time = time.time()
        
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        
        # Evaluate on validation set
        val_loss, val_preds, val_labels = evaluate(model, val_loader, device)
        
        # Calculate validation accuracy
        val_accuracy = accuracy_score(val_labels, val_preds)
        
        # Print results
        time_elapsed = time.time() - start_time
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        print(f"Epoch completed in {time_elapsed:.2f}s")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Return history and best model state
    return history, best_model_state

def plot_learning_curves(history):
    """
    Plot learning curves
    
    Args:
        history (dict): Training history
    """
    plt.figure(figsize=(12, 5))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'])
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels):
    """
    Plot confusion matrix
    
    Args:
        y_true (list): True labels
        y_pred (list): Predicted labels
        labels (list): Class labels
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))
