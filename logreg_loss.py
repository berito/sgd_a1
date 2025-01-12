import numpy as np
# Logistic regression loss function
def logreg_loss(X, y, w):
    """
    Computes the logistic regression loss for the given dataset and weights.
    
    Args:
        X (ndarray): Input data of shape (n_samples, n_features).
        y (ndarray): Labels of shape (n_samples,).
        w (ndarray): Weights of shape (n_features,).
    
    Returns:
        float: The logistic regression loss.
    """
    n = X.shape[0]
    # Compute linear predictions
    z = X.dot(w)
    # Compute the logistic loss
    loss = np.mean(np.log(1 + np.exp(-y * z)))
    return loss