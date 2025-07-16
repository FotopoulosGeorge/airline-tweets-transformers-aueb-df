# # PLS GOTO README.MD FIRST!!

import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

from data_preprocessing import load_and_preprocess_data, create_data_loaders
from model import SentimentClassifier
from training import train_model, evaluate, plot_learning_curves, plot_confusion_matrix

def main():
    try:
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Model and tokenizer name
        model_name = 'distilbert-base-uncased'
        print(f"Using model: {model_name}")
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        print("Random seed set")
        
        # Hyperparameters
        max_length = 128
        batch_size = 32
        learning_rate = 5e-5
        weight_decay = 0.01
        epochs = 6
        print("Hyperparameters set")
        
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
        print("Tokenizer loaded successfully")
        
        print("Creating data loaders...")
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            train_df, val_df, test_df, tokenizer, 
            batch_size=batch_size, max_length=max_length
        )
        print("Data loaders created successfully")
        
        print("Initializing model...")
        # Initialize model
        model = SentimentClassifier(
            model_name=model_name,
            num_classes=num_classes
        )
        print("Model initialized successfully")
        
        # Move model to device
        model.to(device)
        print("Model moved to device")
        
        print("Initializing optimizer and scheduler...")
        # Initialize optimizer with layer-wise learning rate decay
        try:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.transformer.named_parameters()],
                    "lr": learning_rate / 10,
                },
                {
                    "params": [p for n, p in model.classifier.named_parameters()],
                    "lr": learning_rate,
                },
            ]
            print("Optimizer parameters grouped successfully")
            
            optimizer = AdamW(
                optimizer_grouped_parameters,
                weight_decay=weight_decay
            )
            print("Optimizer created successfully")
            
            # Calculate total steps
            total_steps = len(train_loader) * epochs
            
            # Initialize scheduler
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * total_steps),
                num_training_steps=total_steps
            )
            print("Scheduler created successfully")
            
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
                patience=3
            )
            print("Model training completed successfully")
            
            # Load best model state
            model.load_state_dict(best_model_state)
            print("Loaded best model state")
            
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
        except Exception as e:
            print(f"Error in optimizer or training: {e}")
            import traceback
            traceback.print_exc()
            raise  # Re-raise the exception to see the full traceback
            
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()