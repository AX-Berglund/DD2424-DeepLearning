
import numpy as np
import copy

def MiniBatchGD(X, Y, GDparams, init_net, lam):
    """
    Performs Mini-Batch Gradient Descent to train the network.

    Parameters:
      X        : d x N input data matrix (each column is an image)
      Y        : K x N one-hot ground truth labels
      GDparams : Dictionary containing:
                 - "n_batch": Mini-batch size
                 - "eta": Learning rate
                 - "n_epochs": Number of epochs
      init_net : Dictionary containing initial parameters:
                 - "W": K x d weight matrix
                 - "b": K x 1 bias vector
      lam      : Regularization coefficient

    Returns:
      trained_net : Dictionary with the learned parameters:
                    - "W": Trained weight matrix
                    - "b": Trained bias vector
    """
    # Make a deep copy to avoid modifying the original dictionary
    trained_net = copy.deepcopy(init_net)

    # Extract hyperparameters
    n_batch = GDparams["n_batch"]
    eta = GDparams["eta"]
    n_epochs = GDparams["n_epochs"]
    
    N = X.shape[1]  # Total number of samples
    rng = np.random.default_rng(seed=42)  # Ensure reproducibility

    for epoch in range(n_epochs):
        # Shuffle the training data
        shuffled_indices = rng.permutation(N)
        X_shuffled = X[:, shuffled_indices]
        Y_shuffled = Y[:, shuffled_indices]

        # Mini-Batch Gradient Descent
        for j in range(N // n_batch):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch
            Xbatch = X_shuffled[:, j_start:j_end]  # Mini-batch inputs (d x n_batch)
            Ybatch = Y_shuffled[:, j_start:j_end]  # Mini-batch labels (K x n_batch)

            # Forward pass: Compute softmax probabilities
            P = ApplyNetwork(Xbatch, trained_net)

            # Compute gradients
            grads = BackwardPass(Xbatch, Ybatch, P, trained_net, lam)

            # Update parameters using Gradient Descent:
            trained_net["W"] -= eta * grads["W"]
            trained_net["b"] -= eta * grads["b"]

        # Compute and print the loss at the end of each epoch
        P_train = ApplyNetwork(X, trained_net)
        loss = ComputeLoss(P_train, np.argmax(Y, axis=0), trained_net["W"], lam)
        print(f"Epoch {epoch+1}/{n_epochs} - Loss: {loss:.4f}")

    return trained_net
