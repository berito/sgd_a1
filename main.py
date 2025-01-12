# Main function to orchestrate the process
import numpy as np
import pandas as pd
from plot import plot,plot_experiment_results
from evaluate import evaluate
from load_data import load_data
from hyperparamter_experiment import run_experiment

def main():
    """
    Main function to run the evaluation and plot the results.
    """
    # hyperparameters
    # alpha = 0.001
    # sigma = 0.0001
    # epochs = 10

   
    train_file = 'dataset/a9a.train.txt' 
    test_file = 'dataset/a9a.test.txt'    

    # Load training and validation data
    X_train, y_train = load_data(train_file)
    X_val, y_val = load_data(test_file)

    # Print shapes to verify
    print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}, Validation labels shape: {y_val.shape}")
    # Evaluate SGD
    # epochs_x, training_loss, training_error, validation_error = evaluate(
    #     X_train, y_train, X_val, y_val, alpha, sigma, epochs
    # )

    # Plot results
    # plot(epochs_x, training_loss, training_error, validation_error)
    # Hyperparameters to test
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    regularization_constants = [0, 0.0001, 0.001, 0.01]
    momentums = [0.0, 0.5, 0.9]
    batch_sizes = [32, 64, 128]

    # Run experiments
    results = run_experiment(X_train, y_train, X_val, y_val, learning_rates, regularization_constants, momentums,batch_sizes)
    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    df.to_csv("sgd_results.csv", index=False)

    print("Results saved to sgd_results.csv") 
    # Plot results
    # plot_experiment_results(results)
# Run the main function
if __name__ == "__main__":
    main() 