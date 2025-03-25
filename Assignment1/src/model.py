import numpy as np
import copy
from data_loader import LoadBatch
from loss import ComputeLoss
from gradient import BackwardPass
from evaluate import ComputeAccuracy, CompareGradients

# Utility Functions
def softmax(S):
    """Computes column-wise softmax of S, ensuring numerical stability."""
    S_shifted = S - np.max(S, axis=0, keepdims=True)
    exp_S = np.exp(S_shifted)
    return exp_S / np.sum(exp_S, axis=0, keepdims=True)

def ApplyNetwork(X, net):
    """
    Applies the network function: Computes class scores and applies softmax.
    
    Parameters:
      X   : d x N input images (each column is an image)
      net : Dictionary containing network parameters {"W": Weight matrix, "b": Bias vector}
    
    Returns:
      P : K x N matrix of class probabilities after softmax
    """
    return softmax(net["W"] @ X + net["b"])

def MiniBatchGD(X, Y, GDparams, init_net, lam):
    """
    Performs Mini-Batch Gradient Descent to train the network.

    Parameters:
      X        : d x N input data matrix (each column is an image)
      Y        : K x N one-hot encoded ground truth labels
      GDparams : Dictionary containing {"n_batch": batch size, "eta": learning rate, "n_epochs": epochs}
      init_net : Dictionary containing initial parameters {"W": weight matrix, "b": bias vector}
      lam      : Regularization coefficient

    Returns:
      trained_net : Dictionary with the learned parameters {"W": Trained weight matrix, "b": Trained bias vector}
    """
    trained_net = copy.deepcopy(init_net)  # Ensure original parameters remain unchanged
    n_batch, eta, n_epochs = GDparams["n_batch"], GDparams["eta"], GDparams["n_epochs"]
    N = X.shape[1]  # Number of samples
    rng = np.random.default_rng(seed=42)  # Ensure reproducibility

    for epoch in range(n_epochs):
        shuffled_indices = rng.permutation(N)
        X_shuffled, Y_shuffled = X[:, shuffled_indices], Y[:, shuffled_indices]

        for j in range(N // n_batch):
            j_start, j_end = j * n_batch, (j + 1) * n_batch
            Xbatch, Ybatch = X_shuffled[:, j_start:j_end], Y_shuffled[:, j_start:j_end]

            # Forward pass and compute gradients
            P = ApplyNetwork(Xbatch, trained_net)
            grads = BackwardPass(Xbatch, Ybatch, P, trained_net, lam)

            # Update parameters
            trained_net["W"] -= eta * grads["W"]
            trained_net["b"] -= eta * grads["b"]

        # Compute and print loss at the end of each epoch
        loss = ComputeLoss(ApplyNetwork(X, trained_net), np.argmax(Y, axis=0), trained_net["W"], lam)
        print(f"Epoch {epoch+1}/{n_epochs} - Loss: {loss:.4f}")

    return trained_net

def preprocess_data(X_train_raw, X_val_raw, X_test_raw):
    """
    Normalizes the dataset based on training set mean and standard deviation.
    
    Parameters:
      X_train_raw, X_val_raw, X_test_raw: Raw data matrices
    
    Returns:
      X_train, X_val, X_test: Normalized datasets
    """
    X_train_mean = np.mean(X_train_raw, axis=1, keepdims=True)
    X_train_std = np.std(X_train_raw, axis=1, keepdims=True)

    X_train = (X_train_raw - X_train_mean) / X_train_std
    X_val = (X_val_raw - X_train_mean) / X_train_std
    X_test = (X_test_raw - X_train_mean) / X_train_std

    return X_train, X_val, X_test

def initialize_network(K=10, d=3072, seed=42):
    """
    Initializes the network parameters.
    
    Parameters:
      K   : Number of classes
      d   : Input dimension
      seed: Random seed for reproducibility
    
    Returns:
      init_net : Dictionary containing initialized parameters {"W": weight matrix, "b": bias vector}
    """
    rng = np.random.default_rng(seed)
    return {"W": 0.01 * rng.standard_normal(size=(K, d)), "b": np.zeros((K, 1))}

def main():
    # Load and preprocess data
    X_train_raw, Y_train, y_train = LoadBatch(1)
    X_val_raw, Y_val, y_val = LoadBatch(2)
    X_test_raw, Y_test, y_test = LoadBatch("test_batch")

    X_train, X_val, X_test = preprocess_data(X_train_raw, X_val_raw, X_test_raw)

    # Initialize network
    init_net = initialize_network()

    # Forward propagation using Softmax
    P = ApplyNetwork(X_train[:, :20], init_net)

    # Compute loss and accuracy
    loss = ComputeLoss(P, y_train[0][:20], init_net["W"], 0.1)
    accuracy = ComputeAccuracy(P, y_train[0][:20])

    # Compute gradients and compare with PyTorch
    grads_own = BackwardPass(X_train[:, :20], Y_train[:, :20], P, init_net, 0.1)
    grads_torch = BackwardPass(X_train[:, :20], Y_train[:, :20], P, init_net, 0.1)
    CompareGradients(grads_own, grads_torch)

    # Train the network
    GDparams = {"n_batch": 100, "eta": 0.001, "n_epochs": 10}
    trained_net = MiniBatchGD(X_train, Y_train, GDparams, init_net, 0.1)

    # Evaluate the trained network
    P_train, P_val, P_test = ApplyNetwork(X_train, trained_net), ApplyNetwork(X_val, trained_net), ApplyNetwork(X_test, trained_net)

    # Compute accuracy
    print("\nTraining Accuracy:", ComputeAccuracy(P_train, y_train[0]))
    print("Validation Accuracy:", ComputeAccuracy(P_val, y_val[0]))
    print("Testing Accuracy:", ComputeAccuracy(P_test, y_test[0]))

if __name__ == "__main__":
    main()
