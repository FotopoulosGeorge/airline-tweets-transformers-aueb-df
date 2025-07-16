# PLS GO TO README.MD FIRST!!!

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

def load_and_preprocess_data(file_path, test_size=0.2, val_size=0.1, random_state=42):
    """
    Load and preprocess the Airline Tweets dataset
    
    Args:
        file_path (str): Path to the CSV file
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of training data to use for validation
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: train, val, test dataframes and label encoder
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Keep only relevant columns
    df = df[['text', 'airline_sentiment']]
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['sentiment_label'] = label_encoder.fit_transform(df['airline_sentiment'])
    
    # Split into train and test sets
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['sentiment_label']
    )
    
    # Split train into train and validation sets
    train_df, val_df = train_test_split(
        train_df, 
        test_size=val_size/(1-test_size),  # Adjust validation size
        random_state=random_state,
        stratify=train_df['sentiment_label']
    )
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    return train_df, val_df, test_df, label_encoder

class AirlineTweetDataset(Dataset):
    """Dataset class for Airline Tweets"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Initialize the dataset
        
        Args:
            texts (list): List of tweet texts
            labels (list): List of sentiment labels
            tokenizer: HuggingFace tokenizer
            max_length (int): Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loaders(train_df, val_df, test_df, tokenizer, batch_size=16, max_length=128):
    """
    Create PyTorch DataLoaders for train, validation and test sets
    
    Args:
        train_df (DataFrame): Training data
        val_df (DataFrame): Validation data
        test_df (DataFrame): Test data
        tokenizer: HuggingFace tokenizer
        batch_size (int): Batch size
        max_length (int): Maximum sequence length
        
    Returns:
        tuple: train, validation and test DataLoaders
    """
    # Create datasets
    train_dataset = AirlineTweetDataset(
        train_df['text'].values,
        train_df['sentiment_label'].values,
        tokenizer,
        max_length
    )
    
    val_dataset = AirlineTweetDataset(
        val_df['text'].values,
        val_df['sentiment_label'].values,
        tokenizer,
        max_length
    )
    
    test_dataset = AirlineTweetDataset(
        test_df['text'].values,
        test_df['sentiment_label'].values,
        tokenizer,
        max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader
