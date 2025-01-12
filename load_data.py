import numpy as np

def load_data(file_path, num_features=123):
    """
    Loads the Adult dataset from the given file and converts it into a dense numpy matrix.
    Appends an extra feature (bias term) of 1 to the dataset.
    
    Args:
        file_path (str): Path to the dataset file.
        num_features (int): Number of features in the dataset (excluding the bias term).
    
    Returns:
        X (ndarray): Dense feature matrix of shape (n_samples, num_features + 1).
        y (ndarray): Labels vector of shape (n_samples,).
    """
    features = []
    labels = []
    
    # Open and read the dataset file
    with open(file_path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            # First token is the label
            labels.append(int(tokens[0]))
            # Initialize feature vector with zeros
            x = np.zeros(num_features)
            # Populate features from the sparse representation
            for token in tokens[1:]:
                index, value = map(float, token.split(':'))
                x[int(index) - 1] = value  # Convert 1-indexed to 0-indexed
            features.append(x)
    
    # Convert lists to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    # Append the bias term (column of ones)
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    
    return X, y
