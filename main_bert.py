# # PLS GOTO README.MD FIRST!

import torch
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup #to linear de douleve kala
from torch.optim import AdamW

from data_preprocessing import load_and_preprocess_data, create_data_loaders
from model import SentimentClassifier
from training import train_model, evaluate, plot_learning_curves, plot_confusion_matrix

def main():
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model and tokenizer name
    model_name = 'bert-base-uncased'
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Hyperparameters
    max_length = 128
    batch_size = 16
    learning_rate = 3e-5
    weight_decay = 0.01
    epochs = 10
    
    # Training strategy parameters
    freeze_base = True
    gradual_unfreeze = {0: 4, 1: 8} 
    
    print("Loading and preprocessing data...")
    # Load and preprocess data
    train_df, val_df, test_df, label_encoder = load_and_preprocess_data(
        file_path='Tweets.csv',
        test_size=0.2,
        val_size=0.1
    )
    
    # Get class labels
    class_names = label_encoder.classes_
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    
    print("Loading tokenizer...")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Creating data loaders...")
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, val_df, test_df, tokenizer, 
        batch_size=batch_size, max_length=max_length
    )
    
    print("Initializing model...")
    # Initialize model
    model = SentimentClassifier(
        model_name=model_name,
        num_classes=num_classes
    )
    
    # Apply training strategy: freeze base model if needed
    if freeze_base:
        print("Freezing base model...")
        model.freeze_base_model()
    
    # Move model to device
    model.to(device)
    
    print("Initializing optimizer and scheduler...")
    # Initialize optimizer
    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Calculate total steps
    total_steps = len(train_loader) * epochs
    
    # Initialize scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    print("Training model...")
    # Train model
    history, best_model_state = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epochs,
        gradual_unfreeze=gradual_unfreeze
    )
    
    # Load best model state
    model.load_state_dict(best_model_state)
    
    print("Plotting learning curves...")
    # Plot learning curves
    plot_learning_curves(history)
    
    print("Evaluating on test set...")
    # Evaluate on test set
    test_loss, test_preds, test_labels = evaluate(model, test_loader, device)
    
    # Print test results
    test_accuracy = (torch.tensor(test_preds) == torch.tensor(test_labels)).float().mean()
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")
    
    # Plot confusion matrix
    print("Plotting confusion matrix...")
    plot_confusion_matrix(test_labels, test_preds, class_names)
    
    print("Done!")

if __name__ == "__main__":
    main()
