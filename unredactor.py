import pandas as pd
from data_preprocessing import preprocess_data
from feature_extraction import extract_features
from model import train_model, predict
from evaluate import evaluate_model

def main():
    # Load data
    df = pd.read_csv('unredactor.tsv', on_bad_lines='skip', sep='\t', names=['split', 'name', 'context'])
    
    # Preprocess data
    train_data, val_data = preprocess_data(df)
    
    # Extract features
    X_train, y_train = extract_features(train_data)
    X_val, y_val = extract_features(val_data)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Predict on validation set
    y_pred = predict(model, X_val)
    
    # Evaluate model
    evaluate_model(y_val, y_pred)

if __name__ == "__main__":
    main()
#unredactor