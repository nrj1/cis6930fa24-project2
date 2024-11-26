import pandas as pd
from data_preprocessing import preprocess_data
from feature_extraction import extract_features_train, extract_features_val_test
from model import train_model, predict
from evaluate import evaluate_model

def main():
    try:
        # Load data
        df = pd.read_csv('unredactor.tsv', on_bad_lines='skip', sep='\t', names=['split', 'name', 'context'])
        print("DataFrame loaded successfully:")
        print(df.head())
        print(df.columns)
        print(df.dtypes)
    except Exception as e:
        print(f"Error loading unredactor.tsv: {e}")
        return
    
    # Preprocess data
    try:
        train_data, val_data = preprocess_data(df)
        print("Preprocessed training data:")
        print(train_data.head())
        print(train_data.dtypes)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return
    
    # Extract features
    try:
        X_train, y_train, vectorizer = extract_features_train(train_data)
        X_val, y_val = extract_features_val_test(val_data, vectorizer)
        print("Feature extraction completed:")
        #print(X_train.dtypes)
        #print(X_train.columns)
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return
    
    # Train model
    try:
        model = train_model(X_train, y_train)
        print("Model training completed.")
    except Exception as e:
        print(f"Error during model training: {e}")
        return
    
    # Predict on validation set
    
    try:
        y_pred = predict(model, X_val)
        
        print("Prediction on validation set completed.")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return
    
    # Evaluate model
    try:
        evaluate_model(y_val, y_pred)
    except Exception as e:
        print(f"Error during model evaluation: {e}")
    
    # Load test data
    try:
        test_df = pd.read_csv('test.tsv', sep='\t', names=['id', 'context'])
        print("Test data loaded successfully:")
        print(test_df.head())
    except Exception as e:
        print(f"Error loading test.tsv: {e}")
        return
    
    # Preprocess and extract features from test data
    try:
        X_test, _ = extract_features_val_test(test_df, vectorizer, True)  # Adapt extract_features for test data
        print("Test data features extracted successfully.")
    except Exception as e:
        print(f"Error during test feature extraction: {e}")
        return
    
    # Predict using the trained model
    try:
        predicted_names = predict(model, X_test)
        print("Prediction on test data completed.")
    except Exception as e:
        print(f"Error during test prediction: {e}")
        return
    
    # Create submission DataFrame
    try:
        submission_df = pd.DataFrame({'id': test_df['id'], 'name': predicted_names})
        submission_df.to_csv('submission.tsv', sep='\t', index=False)
        print("Submission file created: submission.tsv")
    except Exception as e:
        print(f"Error creating submission file: {e}")

if __name__ == "__main__":
    main()
