import torch
import numpy as np

def ComputeGradsWithTorch(X, y, network_params, lam):
    """
    Computes gradients using PyTorch with L2 regularization.

    Parameters:
      X              : d x N numpy array (input data)
      y              : 1D numpy array of length N (true class indices)
      network_params : Dictionary with keys:
                       - "W" : K x d weight matrix
                       - "b" : K x 1 bias vector
      lam            : Regularization coefficient (lambda)
    
    Returns:
      grads: Dictionary with computed gradients:
             - "W": K x d gradient matrix
             - "b": K x 1 gradient vector
    """
    # Convert numpy arrays to PyTorch tensors
    Xt = torch.from_numpy(X).float()

    # Create tensors for weights and biases that require gradients
    W = torch.tensor(network_params['W'], dtype=torch.float32, requires_grad=True)
    b = torch.tensor(network_params['b'], dtype=torch.float32, requires_grad=True)    

    N = X.shape[1]  # Number of samples

    # Compute raw scores: S = W * X + b
    scores = torch.matmul(W, Xt) + b

    # Apply softmax
    apply_softmax = torch.nn.Softmax(dim=0)
    P = apply_softmax(scores)

    # Compute cross-entropy loss
    loss = torch.mean(-torch.log(P[y, np.arange(N)]))

    # Compute total cost (loss + L2 regularization)
    cost = loss + lam * torch.sum(W * W)

    # Compute gradients w.r.t cost
    cost.backward()

    # Extract computed gradients and convert to numpy
    grads = {
        "W": W.grad.numpy(),
        "b": b.grad.numpy()
    }

    return grads
