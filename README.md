# Unredaction Project

This project aims to unredact names from redacted text using machine learning techniques. It processes a dataset of redacted texts, extracts features, trains a model, and predicts the redacted names.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Key Components](#key-components)
5. [Important Functions](#important-functions)

## Project Structure

The project consists of the following Python files:

- `unredactor.py`: Main script that orchestrates the entire process
- `data_preprocessing.py`: Handles data preprocessing
- `feature_extraction.py`: Extracts features from the text data
- `model.py`: Contains model training and prediction functions
- `evaluate.py`: Evaluates the model's performance

## Installation

1. Ensure you have Python 3.10 installed.
2. Install the required packages:

```bash
pip install pandas scikit-learn nltk numpy
```

3. Download required NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
```

## Usage

1. Place your input data files (`unredactor.tsv` and `test.tsv`) in the project directory.
2. Run the main script:

```bash
python unredactor.py
```

3. The script will process the data, train the model, evaluate its performance, and generate a `submission.tsv` file with predictions for the test data.

## Key Components

1. **Data Preprocessing**: The `preprocess_data` function in `data_preprocessing.py` splits the data into training and validation sets.

2. **Feature Extraction**: The `extract_features_train` and `extract_features_val_test` functions in `feature_extraction.py` extract features from the text data, including n-grams, redaction length, and POS/NER tags.

3. **Model Training**: The `train_model` function in `model.py` trains a Logistic Regression model on the extracted features.

4. **Prediction**: The `predict` function in `model.py` uses the trained model to make predictions on new data.

5. **Evaluation**: The `evaluate_model` function in `evaluate.py` calculates and prints precision, recall, and F1-score for the model's predictions.

## Important Functions

1. `main()` in `unredactor.py`: Orchestrates the entire unredaction process, including data loading, preprocessing, feature extraction, model training, prediction, and evaluation.

2. `extract_features_train(data)` in `feature_extraction.py`: Extracts features from the training data, including n-grams, redaction length, and POS/NER tags.

3. `extract_features_val_test(data, vectorizer, is_test=False)` in `feature_extraction.py`: Extracts features from validation or test data using the pre-fitted vectorizer.

4. `train_model(X_train, y_train)` in `model.py`: Trains a Logistic Regression model on the given features and labels.

5. `predict(model, X_val)` in `model.py`: Makes predictions using the trained model on new data.

6. `evaluate_model(y_true, y_pred)` in `evaluate.py`: Calculates and prints precision, recall, and F1-score for the model's predictions.

Note: The project uses a Logistic Regression model with L2 regularization for name prediction. The feature extraction process includes n-grams, redaction length, and basic NLP features (POS tags and Named Entity Recognition).

## Output
