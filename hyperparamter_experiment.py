import numpy as np
from logreg_error import logreg_error
from logreg_loss import logreg_loss
import logreg_sgd
def run_experiment(X_train, y_train, X_val, y_val, learning_rates, regularization_constants, momentums,batch_sizes):
    results = []
    # Total configurations to track progress
    total_configurations = len(learning_rates) * len(regularization_constants) * len(momentums) * len(batch_sizes)
    current_configuration = 0

    for alpha in learning_rates:
        for sigma in regularization_constants:
            for momentum in momentums:
                for batch_size in batch_sizes:
                    current_configuration += 1

                    # Initialize weights
                    w = np.zeros(X_train.shape[1])
                    n_train = X_train.shape[0]
                    epochs = 10
                    T_per_epoch = n_train // batch_size  # Number of batches per epoch

                    # Metrics to track
                    training_loss = []
                    training_error = []
                    validation_error = []

                    # SGD with momentum
                    velocity = np.zeros_like(w)
                    for epoch in range(epochs):
                        # Shuffle the dataset at the start of each epoch
                        indices = np.arange(n_train)
                        np.random.shuffle(indices)
                        X_train_shuffled = X_train[indices]
                        y_train_shuffled = y_train[indices]

                        for batch_start in range(0, n_train, batch_size):
                            batch_end = min(batch_start + batch_size, n_train)
                            X_batch = X_train_shuffled[batch_start:batch_end]
                            y_batch = y_train_shuffled[batch_start:batch_end]

                            # Compute gradient for the mini-batch
                            gradients = np.zeros_like(w)
                            for xi, yi in zip(X_batch, y_batch):
                                gradients += -(yi * xi) / (1 + np.exp(yi * np.dot(w, xi)))

                            # Add regularization term
                            gradients = gradients / batch_size + 2 * sigma * w

                            # Update rule with momentum
                            velocity = momentum * velocity - alpha * gradients
                            w += velocity

                        # Record metrics
                        training_loss.append(logreg_loss(X_train, y_train, w))
                        training_error.append(logreg_error(X_train, y_train, w))
                        validation_error.append(logreg_error(X_val, y_val, w))

                    # Save results
                    results.append({
                        "alpha": alpha,
                        "sigma": sigma,
                        "momentum": momentum,
                        "batch_size": batch_size,
                        "training_loss": training_loss[-1],  # Save the final loss
                        "training_error": training_error[-1],  # Save the final error
                        "validation_error": validation_error[-1]  # Save the final validation error
                    })

                    # Print progress message
                    print(f"Completed {current_configuration}/{total_configurations}: "
                        f"alpha={alpha}, sigma={sigma}, momentum={momentum}, batch_size={batch_size}")

    return results