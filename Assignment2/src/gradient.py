import numpy as np
def BackwardPass(X, Y, P, network, lam):
    """
    Computes gradients of the cost function J with respect to W and b.

    Parameters:
      X      : d x N input data matrix (each column is an image)
      Y      : K x N one-hot ground truth labels
      P      : K x N softmax probabilities from ApplyNetwork
      network: Dictionary containing model parameters:
               - "W": K x d weight matrix
               - "b": K x 1 bias vector
      lam    : Regularization coefficient

    Returns:
      grads: Dictionary with gradients
             - "W": K x d gradient matrix
             - "b": K x 1 gradient vector
    """
    N = X.shape[1]  # Number of samples in mini-batch

    # Compute gradient of the loss w.r.t. S (logits)
    G = P - Y  # (K x N)

    # Compute gradients
    grad_W = (1 / N) * (G @ X.T) + 2 * lam * network["W"]  # (K x d)
    grad_b = (1 / N) * np.sum(G, axis=1, keepdims=True)     # (K x 1)

    # Store gradients in a dictionary
    grads = {
        "W": grad_W,
        "b": grad_b
    }
    
    return grads
