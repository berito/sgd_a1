import numpy as np
# Stochastic gradient descent for logistic regression
def logreg_sgd(X, y, w, alpha, sigma, T):
    """
    Runs stochastic gradient descent for logistic regression.
    
    Args:
        X (ndarray): Input data of shape (n_samples, n_features).
        y (ndarray): Labels of shape (n_samples,).
        w (ndarray): Initial weights of shape (n_features,).
        alpha (float): Learning rate.
        sigma (float): L2 regularization constant.
        T (int): Number of steps to run SGD.
    
    Returns:
        ndarray: The updated weights after running SGD.
    """
    n = X.shape[0]
    for t in range(T):
        # Select a random training example
        i = np.random.randint(n)
        xi = X[i]
        yi = y[i]
        
        # Compute the gradient of the logistic loss
        gradient = -(yi * xi) / (1 + np.exp(yi * np.dot(w, xi))) + 2 * sigma * w
        
        # Update weights using the gradient
        w -= alpha * gradient
    
    return w