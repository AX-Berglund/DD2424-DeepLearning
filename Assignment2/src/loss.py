import numpy as np


def ComputeLoss(P, y, W, lam):
    """
    Computes the cost function J using softmax probabilities.
    
    Parameters:
      P   : K x N matrix of softmax probabilities
      y   : 1D array of length N with ground truth class indices (integers 0 to 9)
      W   : K x d weight matrix (needed for regularization term)
      lam : Regularization coefficient (float)
    
    Returns:
      J : Scalar value representing the total loss (cross-entropy + regularization)
    """
    # Number of training examples
    N = P.shape[1]

    # Pick correct class probabilities for each example using indexing
    # Example P[y=5,Sample1] is the probability of class 5 for the first example
    log_probs = -np.log(P[y, np.arange(N)] + 1e-15)  # Avoid log(0) 
    
    # Cross-entropy loss (mean over all examples)
    cross_entropy_loss = np.mean(log_probs)

    # Regularization term (L2 penalty on W)
    reg_term = lam * np.sum(W**2)

    # Total cost
    J = cross_entropy_loss + reg_term
    
    return J