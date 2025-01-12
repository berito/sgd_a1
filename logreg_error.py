import numpy as np
# Logistic regression error function
def logreg_error(X, y, w):
    """
    Computes the classification error for the given dataset and weights.
    
    Args:
        X (ndarray): Input data of shape (n_samples, n_features).
        y (ndarray): Labels of shape (n_samples,).
        w (ndarray): Weights of shape (n_features,).
    
    Returns:
        float: The fraction of incorrectly classified samples (error rate).
    """
    predictions = np.sign(X.dot(w))
    error = np.mean(predictions != y)
    return error