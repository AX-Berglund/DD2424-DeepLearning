import numpy as np

def init_rng(seed=400):
    """Initialize the random number generator."""
    return np.random.RandomState(seed)

def compute_loss(Y, P):
    """
    Compute the cross-entropy loss.
    
    Args:
        Y: One-hot encoded target characters, shape (K, seq_length)
        P: Predicted probabilities, shape (K, seq_length)
    
    Returns:
        loss: Scalar cross-entropy loss
    """
    # Small constant to avoid log(0)
    eps = 1e-10
    
    # Cross-entropy loss
    loss = -np.sum(Y * np.log(P + eps)) / P.shape[1]
    return loss

def sample_next_char(p, rng):
    """
    Sample the next character based on the probability distribution.
    
    Args:
        p: Probability distribution, shape (K, 1)
        rng: Random number generator
    
    Returns:
        index: Sampled index
    """
    cp = np.cumsum(p, axis=0)
    a = rng.uniform(size=1)
    index = np.argmax(cp - a > 0)
    return index

def sample_with_temperature(p, T=1.0):
    """
    Apply temperature to softmax probabilities.
    
    Args:
        p: Original logits before softmax, shape (K, 1)
        T: Temperature parameter (lower = more conservative)
    
    Returns:
        p_new: New probability distribution
    """
    if T == 1.0:
        return p
    
    # Convert probabilities back to logits
    log_p = np.log(p + 1e-10)
    # Apply temperature
    log_p_temp = log_p / T
    # Re-apply softmax
    p_temp = np.exp(log_p_temp)
    p_new = p_temp / np.sum(p_temp, axis=0, keepdims=True)
    
    return p_new

def nucleus_sampling(p, theta=0.9):
    """
    Implement nucleus sampling as described in the paper
    "The Curious Case of Neural Text Degeneration".
    
    Args:
        p: Probability distribution, shape (K, 1)
        theta: Threshold for cumulative probability
    
    Returns:
        p_new: New probability distribution
    """
    # Sort probabilities in descending order
    sorted_indices = np.argsort(p, axis=0)[::-1]
    sorted_p = p[sorted_indices]
    
    # Find cumulative sum
    cumsum = np.cumsum(sorted_p)
    
    # Find the smallest k such that cumsum[k] >= theta
    k = np.argmax(cumsum >= theta) + 1
    
    # Create a new distribution with only the top k probabilities
    p_new = np.zeros_like(p)
    top_indices = sorted_indices[:k]
    p_new[top_indices] = p[top_indices]
    
    # Normalize
    p_new = p_new / np.sum(p_new, axis=0, keepdims=True)
    
    return p_new