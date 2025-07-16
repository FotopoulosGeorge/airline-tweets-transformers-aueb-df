# # PLS GOTO README.MD FIRST!!!

import torch
import torch.nn as nn
from transformers import AutoModel

class SentimentClassifier(nn.Module):
    
    def __init__(self, model_name, num_classes, dropout_rate=0.1):
        """
        Initialize the model
        
        Args:
            model_name (str): Name of the pretrained model
            num_classes (int): Number of sentiment classes
            dropout_rate (float): Dropout rate
        """
        super(SentimentClassifier, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        # Get embedding dimension from model config
        hidden_size = self.transformer.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids (tensor): Token ids
            attention_mask (tensor): Attention mask
            
        Returns:
            tensor: Logits for each class
        """
        # Pass input through transformer model
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get the [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout and classification layer
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def freeze_base_model(self):
        """Freeze the parameters of the base transformer model"""
        for param in self.transformer.parameters():
            param.requires_grad = False
            
    def unfreeze_base_model(self):
        """Unfreeze the parameters of the base transformer model"""
        for param in self.transformer.parameters():
            param.requires_grad = True
            
    def gradual_unfreeze(self, num_layers_to_unfreeze):
        """
        Gradually unfreeze layers from top to bottom
        
        Args:
            num_layers_to_unfreeze (int): Number of layers to unfreeze
        """
        # Freeze all layers first
        self.freeze_base_model()
        
        # Unfreeze classifier
        for param in self.classifier.parameters():
            param.requires_grad = True
            
        # Get list of transformer layers
        if hasattr(self.transformer, 'encoder'):
            layers = self.transformer.encoder.layer
        elif hasattr(self.transformer, 'layer'):
            layers = self.transformer.layer
        else:
            print("Model structure not recognized for gradual unfreezing")
            return
        
        # Unfreeze specified number of layers from the top
        total_layers = len(layers)
        for i in range(min(num_layers_to_unfreeze, total_layers)):
            for param in layers[total_layers - 1 - i].parameters():
                param.requires_grad = True
        
        # Print unfrozen layers
        print(f"Unfrozen {min(num_layers_to_unfreeze, total_layers)} layers from the top")
