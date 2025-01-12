import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Function to plot results
def plot(epochs_x, training_loss, training_error, validation_error,output_file="sgd_results.png"):
    """
    Plots the training loss, training error, and validation error over epochs.
    
    Args:
        epochs_x (ndarray): Array of epoch indices.
        training_loss (list): List of training loss values.
        training_error (list): List of training error values.
        validation_error (list): List of validation error values.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_x, training_loss, label="Training Loss")
    plt.plot(epochs_x, training_error, label="Training Error")
    plt.plot(epochs_x, validation_error, label="Validation Error")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.title("SGD Evaluation: Loss and Errors")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(output_file)  # Save the plot to a file
    print(f"Plot saved to {output_file}")

def plot_experiment_results(results,metric="validation_error",output_file="hyperparameter_results.png"):
    # for result in results:
    #     plt.figure(figsize=(10, 6))
    #     epochs = np.arange(len(result["training_loss"]))
    #     plt.plot(epochs, result["training_loss"], label="Training Loss")
    #     plt.plot(epochs, result["training_error"], label="Training Error")
    #     plt.plot(epochs, result["validation_error"], label="Validation Error")
    #     plt.title(f"Results: alpha={result['alpha']}, sigma={result['sigma']}, momentum={result['momentum']}")
    #     plt.xlabel("Epochs")
    #     plt.ylabel("Metrics")
    #     plt.legend()
    #     plt.grid(True)
    data = []
    for result in results:
        alpha = result["alpha"]
        sigma = result["sigma"]
        momentum = result["momentum"]
        final_metric = result[metric][-1]  # Take the last epoch's value
        data.append({"alpha": alpha, "sigma": sigma, "momentum": momentum, metric: final_metric})
    
    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Grouped bar plot
    plt.figure(figsize=(12, 6))
    for momentum in df["momentum"].unique():
        subset = df[df["momentum"] == momentum]
        plt.bar(
            x=subset["alpha"] + subset["sigma"] / 10 + momentum / 100,
            height=subset[metric],
            label=f"Momentum={momentum}",
            width=0.02
        )
    
    plt.title(f"Bar Plot of {metric.capitalize()} for Different Hyperparameters")
    plt.xlabel("Learning Rate and Regularization")
    plt.ylabel(metric.replace("_", " ").capitalize())
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(output_file)  # Save the plot to a file
    print(f"Plot saved to {output_file}")