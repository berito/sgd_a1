import numpy as np
from logreg_error import logreg_error
from logreg_loss import logreg_loss 
from logreg_sgd import logreg_sgd
# Function to evaluate SGD
def evaluate(X_train, y_train, X_val, y_val, alpha, sigma, epochs):
    """
    Evaluates SGD using the pre-defined `logreg_sgd` function for training.
    
    Args:
        X_train (ndarray): Training data of shape (n_train_samples, n_features).
        y_train (ndarray): Training labels of shape (n_train_samples,).
        X_val (ndarray): Validation data of shape (n_val_samples, n_features).
        y_val (ndarray): Validation labels of shape (n_val_samples,).
        alpha (float): Step size (learning rate).
        sigma (float): Regularization constant.
        epochs (int): Number of epochs to train.

    Returns:
        tuple: (epochs_x, training_loss, training_error, validation_error)
            epochs_x: Array of epoch indices.
            training_loss: List of training loss values.
            training_error: List of training error values.
            validation_error: List of validation error values.
    """
    n_train = X_train.shape[0]
    T_per_epoch = n_train  # Iterations per epoch
    w = np.zeros(X_train.shape[1])  # Initial weights

    # Metrics to track
    training_loss = []
    training_error = []
    validation_error = []

    # Initial metrics
    training_loss.append(logreg_loss(X_train, y_train, w))
    training_error.append(logreg_error(X_train, y_train, w))
    validation_error.append(logreg_error(X_val, y_val, w))

    # SGD training
    for epoch in range(epochs):
        # Use the existing logreg_sgd function for one epoch of updates
        w = logreg_sgd(X_train, y_train, w, alpha, sigma, T_per_epoch)

        # Metrics after each epoch
        training_loss.append(logreg_loss(X_train, y_train, w))
        training_error.append(logreg_error(X_train, y_train, w))
        validation_error.append(logreg_error(X_val, y_val, w))

    # Prepare epoch indices
    epochs_x = np.arange(0, epochs + 1)
    return epochs_x, training_loss, training_error, validation_error
