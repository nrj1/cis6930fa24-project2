import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    # Split data into train and validation sets
    train_data = df[df['split'] == 'training']
    val_data = df[df['split'] == 'validation']
    
    return train_data, val_data