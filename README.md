# Fine-Tuning Transformer Models for Airline Tweet Sentiment Analysis

A comprehensive sentiment analysis project implementing and comparing three transformer models (BERT, RoBERTa, and DistilBERT) for classifying airline customer tweets. Built for AUEB's AI/ML Data Factory course.

You can view the notebook here [![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?logo=Jupyter)](https://nbviewer.org/github/FotopoulosGeorge/airline-tweets-transformers-aueb-df/blob/main/transformer_models.ipynb)

## Project Structure

- `data_preprocessing.py`: Contains functions for data preprocessing and preparation
- `model.py`: Defines the `SentimentClassifier` class architecture
- `training.py`: Contains functions for model training, evaluation, and performance visualization
- `main_bert.py`: BERT model implementation
- `main_roberta.py`: RoBERTa model implementation  
- `main_distilbert.py`: DistilBERT model implementation
- `transformer_models.ipynb`: Main notebook that orchestrates all components

## Model Architectures

### BERT
- Uses `bert-base-uncased` pre-trained model
- Training Configuration:
  - Progressive unfreezing strategy (4 layers at epoch 0, additional 4 layers later)
  - AdamW optimizer with learning rate 3e-5
  - Cosine scheduler with warmup (significantly outperformed linear scheduler)
  - Early unfreezing was necessary for this dataset to achieve proper learning

### RoBERTa 
- Uses `roberta-base` pre-trained model
- Training Configuration:
  - No layer freezing applied
  - Lower learning rate (1e-5) for stable training
  - Cosine scheduler with warmup
  - Higher dropout rate (0.2) for regularization

### DistilBERT 
- Uses `distilbert-base-uncased` pre-trained model
- Training Configuration:
  - Layer-wise learning rate decay implementation
  - Higher learning rate (5e-5) with larger batch size
  - 10% warmup steps for gradual learning rate increase

## Technical Implementation

### Data Processing
- Comprehensive text preprocessing pipeline
- Sentiment label encoding and class balancing
- Train/validation/test splits with stratification

### Training Strategy
- Comparative analysis across three transformer architectures
- Hyperparameter tuning for each model type
- Performance visualization and metrics tracking
- GPU acceleration with T4 (approximate runtime: 45 minutes)

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion matrices for detailed performance analysis
- Learning curves and loss visualization

## Usage

1. Upload all files including `Tweets.csv` to Google Colab
2. Run `transformer_models.ipynb` to execute the complete pipeline
3. Compare results across all three models

## Results & Insights

The project demonstrates the effectiveness of different transformer architectures on sentiment classification tasks, with particular attention to:
- The impact of unfreezing strategies on model performance
- Learning rate optimization for different model sizes
- The trade-offs between model complexity and training efficiency

## Academic Context
**Course**: AI/ML Data Factory  
**Institution**: Athens University of Economics and Business (AUEB)  
**Focus**: Advanced transformer fine-tuning and comparative analysis

---

*This project showcases practical implementation of state-of-the-art transformer models for real-world sentiment analysis applications.*
