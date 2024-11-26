from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train):
    """
    Train a Logistic Regression model.
    """
    # Initialize the Logistic Regression model
    model = LogisticRegression(
        penalty='l2',          # L2 regularization
        solver='lbfgs',        # Solver suitable for small-to-medium datasets
        max_iter=1000,         # Maximum iterations to ensure convergence
        random_state=42        # For reproducibility
    )
    # Train the model
    model.fit(X_train, y_train)
    return model

def predict(model, X_val):
    """
    Predict using the trained model.
    """
    print(model.predict(X_val))
    return model.predict(X_val)
