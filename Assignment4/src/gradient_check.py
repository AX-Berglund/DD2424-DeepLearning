import numpy as np
import torch
import torch.nn.functional as F
from src.torch_grads import ComputeGradsWithTorch

def compute_numerical_gradient(X, Y, h0, RNN, param_name, h=1e-6):
    """
    Compute numerical gradient for a parameter in the RNN.
    
    Args:
        X: Input data, shape (K, seq_length)
        Y: Target data, shape (K, seq_length)
        h0: Initial hidden state, shape (m, 1)
        RNN: Dictionary containing the RNN parameters
        param_name: Name of the parameter to compute the gradient for
        h: Step size for numerical gradient
    
    Returns:
        num_grad: Numerical gradient for the parameter
    """
    # Get the parameter
    param = RNN[param_name]
    
    # Initialize numerical gradient
    num_grad = np.zeros_like(param)
    
    # Compute numerical gradient by perturbing each element
    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        
        # Save the original value
        old_val = param[idx]
        
        # Compute loss with parameter + h
        param[idx] = old_val + h
        P1, _, _, _ = forward_pass(X, h0, RNN)
        loss1 = compute_loss(Y, P1)
        
        # Compute loss with parameter - h
        param[idx] = old_val - h
        P2, _, _, _ = forward_pass(X, h0, RNN)
        loss2 = compute_loss(Y, P2)
        
        # Restore the original value
        param[idx] = old_val
        
        # Compute numerical gradient
        num_grad[idx] = (loss1 - loss2) / (2 * h)
        
        it.iternext()
    
    return num_grad

def forward_pass(X, h0, RNN):
    """
    Forward pass through the RNN.
    
    Args:
        X: Input data, shape (K, seq_length)
        h0: Initial hidden state, shape (m, 1)
        RNN: Dictionary containing the RNN parameters
    
    Returns:
        P: Output probabilities, shape (K, seq_length)
        h: Hidden states, shape (m, seq_length+1)
        a: Pre-activation hidden states, shape (m, seq_length)
        o: Pre-softmax outputs, shape (K, seq_length)
    """
    W = RNN['W']
    U = RNN['U']
    V = RNN['V']
    b = RNN['b']
    c = RNN['c']
    
    seq_length = X.shape[1]
    m = W.shape[0]
    K = V.shape[0]
    
    # Initialize arrays to store intermediate values
    a = np.zeros((m, seq_length))
    h = np.zeros((m, seq_length + 1))
    o = np.zeros((K, seq_length))
    P = np.zeros((K, seq_length))
    
    # Set the initial hidden state
    h[:, 0:1] = h0
    
    # Forward pass through time
    for t in range(seq_length):
        # Input to the hidden layer
        a[:, t:t+1] = np.dot(W, h[:, t:t+1]) + np.dot(U, X[:, t:t+1]) + b
        
        # Hidden state
        h[:, t+1:t+2] = np.tanh(a[:, t:t+1])
        
        # Output layer
        o[:, t:t+1] = np.dot(V, h[:, t+1:t+2]) + c
        
        # Softmax probabilities
        P[:, t:t+1] = np.exp(o[:, t:t+1]) / np.sum(np.exp(o[:, t:t+1]))
    
    return P, h, a, o

def backward_pass(X, Y, P, h, a, RNN):
    """
    Backward pass through the RNN.
    
    Args:
        X: Input data, shape (K, seq_length)
        Y: Target data, shape (K, seq_length)
        P: Output probabilities from forward pass, shape (K, seq_length)
        h: Hidden states from forward pass, shape (m, seq_length+1)
        a: Pre-activation hidden states from forward pass, shape (m, seq_length)
        RNN: Dictionary containing the RNN parameters
    
    Returns:
        grads: Dictionary of gradients for all parameters
    """
    W = RNN['W']
    U = RNN['U']
    V = RNN['V']
    
    seq_length = X.shape[1]
    m = W.shape[0]
    K = V.shape[0]
    
    # Initialize gradients
    dU = np.zeros_like(U)
    dW = np.zeros_like(W)
    dV = np.zeros_like(V)
    db = np.zeros_like(RNN['b'])
    dc = np.zeros_like(RNN['c'])
    
    # Initialize gradient of hidden state
    dh_next = np.zeros((m, 1))
    
    # Backward pass through time
    for t in reversed(range(seq_length)):
        # Gradient of the output
        do = P[:, t:t+1] - Y[:, t:t+1]
        
        # Gradient of V and c
        dV += np.dot(do, h[:, t+1:t+2].T)
        dc += do
        
        # Gradient of hidden state
        dh = np.dot(V.T, do) + dh_next
        
        # Gradient of tanh
        da = (1 - np.square(h[:, t+1:t+2])) * dh
        
        # Gradient of W, U, and b
        db += da
        dW += np.dot(da, h[:, t:t+1].T)
        dU += np.dot(da, X[:, t:t+1].T)
        
        # Gradient for next iteration
        dh_next = np.dot(W.T, da)
    
    # Scale gradients by sequence length (for consistency with loss calculation)
    scale_factor = 1.0 / seq_length
    grads = {
        'U': dU * scale_factor,
        'W': dW * scale_factor,
        'V': dV * scale_factor,
        'b': db * scale_factor,
        'c': dc * scale_factor
    }
    
    return grads

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

def ComputeGradsWithTorch(X, y, h0, RNN):
    """
    Compute gradients using PyTorch's automatic differentiation.
    
    Args:
        X: Input data, shape (d, tau)
        y: Target indices, shape (tau,)
        h0: Initial hidden state, shape (m, 1)
        RNN: Dictionary containing the RNN parameters
    
    Returns:
        grads: Dictionary of gradients for all parameters
    """
    tau = X.shape[1]  # number of time steps

    Xt = torch.from_numpy(X)
    ht = torch.from_numpy(h0)

    torch_network = {}
    for kk in RNN.keys():
        torch_network[kk] = torch.tensor(RNN[kk], requires_grad=True)

    # Give informative names to these torch classes        
    apply_tanh = torch.nn.Tanh()
    apply_softmax = torch.nn.Softmax(dim=0) 
    
    # Create an empty tensor to store the hidden vector at each timestep
    Hs = torch.empty(h0.shape[0], X.shape[1], dtype=torch.float64)
    
    hprev = ht
    for t in range(tau):
        # Equation (1): Compute the hidden scores
        a = torch.matmul(torch_network['W'], hprev) + torch.matmul(torch_network['U'], Xt[:, t:t+1]) + torch_network['b']
        
        # Equation (2): Apply tanh activation
        hnext = apply_tanh(a)
        
        # Store the hidden state
        Hs[:, t] = hnext.squeeze(1)
        
        # Update hprev for the next time step
        hprev = hnext

    # Equation (3): Compute the output scores
    Os = torch.matmul(torch_network['V'], Hs) + torch_network['c']
    
    # Equation (4): Apply softmax
    P = apply_softmax(Os)    
    
    # Compute the loss (cross-entropy)
    loss = torch.mean(-torch.log(P[y, torch.arange(tau)]))
    
    # Compute the backward pass relative to the loss and the named parameters 
    loss.backward()

    # Extract the computed gradients and make them numpy arrays
    grads = {}
    for kk in RNN.keys():
        grads[kk] = torch_network[kk].grad.numpy()

    return grads

def torch_gradient_check(book_data, char_to_ind):
    """
    Check the gradient computation using PyTorch's automatic differentiation.
    
    Args:
        book_data: String containing the book text
        char_to_ind: Dictionary mapping characters to indices
    
    Returns:
        is_correct: Boolean indicating if the gradient computation is correct
    """
    # Parameters
    m = 10  # Use a smaller hidden size for gradient checking
    seq_length = 25
    K = len(char_to_ind)
    
    # Get input data
    X_chars = book_data[:seq_length]
    Y_chars = book_data[1:seq_length+1]
    
    # Convert to one-hot encoding
    X = np.zeros((K, seq_length))
    Y = np.zeros((K, seq_length))
    
    # Target indices (not one-hot)
    y = np.zeros(seq_length, dtype=int)
    
    for t, char in enumerate(X_chars):
        X[char_to_ind[char], t] = 1
    
    for t, char in enumerate(Y_chars):
        Y[char_to_ind[char], t] = 1
        y[t] = char_to_ind[char]
    
    # Initialize RNN parameters
    rng = np.random.RandomState(400)
    RNN = {
        'b': np.zeros((m, 1)),
        'c': np.zeros((K, 1)),
        'U': (1/np.sqrt(2*K)) * rng.standard_normal(size=(m, K)),
        'W': (1/np.sqrt(2*m)) * rng.standard_normal(size=(m, m)),
        'V': (1/np.sqrt(m)) * rng.standard_normal(size=(K, m))
    }
    
    # Initial hidden state
    h0 = np.zeros((m, 1))
    
    # Forward and backward pass
    P, h, a, o = forward_pass(X, h0, RNN)
    grads = backward_pass(X, Y, P, h, a, RNN)
    
    # Get gradients using PyTorch
    torch_grads = ComputeGradsWithTorch(X, y, h0, RNN)
    
    # Compare gradients
    rel_error_W = np.linalg.norm(grads['W'] - torch_grads['W']) / (np.linalg.norm(grads['W']) + np.linalg.norm(torch_grads['W']))
    rel_error_U = np.linalg.norm(grads['U'] - torch_grads['U']) / (np.linalg.norm(grads['U']) + np.linalg.norm(torch_grads['U']))
    rel_error_V = np.linalg.norm(grads['V'] - torch_grads['V']) / (np.linalg.norm(grads['V']) + np.linalg.norm(torch_grads['V']))
    rel_error_b = np.linalg.norm(grads['b'] - torch_grads['b']) / (np.linalg.norm(grads['b']) + np.linalg.norm(torch_grads['b']))
    rel_error_c = np.linalg.norm(grads['c'] - torch_grads['c']) / (np.linalg.norm(grads['c']) + np.linalg.norm(torch_grads['c']))
    
    print(f"Relative error for W: {rel_error_W}")
    print(f"Relative error for U: {rel_error_U}")
    print(f"Relative error for V: {rel_error_V}")
    print(f"Relative error for b: {rel_error_b}")
    print(f"Relative error for c: {rel_error_c}")
    
    # Check if the relative error is small enough
    threshold = 1e-5
    is_correct = all(err < threshold for err in [rel_error_W, rel_error_U, rel_error_V, rel_error_b, rel_error_c])
    
    if not is_correct:
        print(f"\nPyTorch gradient check failed with threshold {threshold}")
        print("This could be due to numerical precision issues or different implementations.")
        print("Try using a smaller network (m=5) or increasing the threshold.")
    
    return is_correct

def check_gradients(book_data, char_to_ind):
    """
    Check the analytic gradients against numerical gradients.
    
    Args:
        book_data: String containing the book text
        char_to_ind: Dictionary mapping characters to indices
    
    Returns:
        is_correct: Boolean indicating if the gradient computation is correct
    """
    # Parameters
    m = 5  # Use a smaller hidden size for gradient checking
    seq_length = 25
    K = len(char_to_ind)
    
    # Get input data
    X_chars = book_data[:seq_length]
    Y_chars = book_data[1:seq_length+1]
    
    # Convert to one-hot encoding
    X = np.zeros((K, seq_length))
    Y = np.zeros((K, seq_length))
    
    for t, char in enumerate(X_chars):
        X[char_to_ind[char], t] = 1
    
    for t, char in enumerate(Y_chars):
        Y[char_to_ind[char], t] = 1
    
    # Initialize RNN parameters
    rng = np.random.RandomState(400)
    RNN = {
        'b': np.zeros((m, 1)),
        'c': np.zeros((K, 1)),
        'U': (1/np.sqrt(2*K)) * rng.standard_normal(size=(m, K)),
        'W': (1/np.sqrt(2*m)) * rng.standard_normal(size=(m, m)),
        'V': (1/np.sqrt(m)) * rng.standard_normal(size=(K, m))
    }
    
    # Initial hidden state
    h0 = np.zeros((m, 1))
    
    # Forward and backward pass
    P, h, a, o = forward_pass(X, h0, RNN)
    grads = backward_pass(X, Y, P, h, a, RNN)
    
    # Compute numerical gradients
    num_grads = {}
    for param_name in ['W', 'U', 'V', 'b', 'c']:
        print(f"Computing numerical gradient for {param_name}...")
        num_grads[param_name] = compute_numerical_gradient(X, Y, h0, RNN, param_name)
    
    # Compare gradients
    rel_errors = {}
    for param_name in ['W', 'U', 'V', 'b', 'c']:
        rel_error = np.linalg.norm(grads[param_name] - num_grads[param_name]) / (np.linalg.norm(grads[param_name]) + np.linalg.norm(num_grads[param_name]))
        rel_errors[param_name] = rel_error
        print(f"Relative error for {param_name}: {rel_error}")
    
    # Check if the relative error is small enough
    threshold = 1e-5  # Increased threshold for numerical stability
    is_correct = all(err < threshold for err in rel_errors.values())
    
    if not is_correct:
        print(f"\nGradient check failed with threshold {threshold}")
        print("This could be due to numerical precision issues.")
        print("Try using a smaller network (m=5) or increasing the threshold.")
    
    return is_correct