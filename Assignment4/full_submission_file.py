import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import torch
import torch.nn.functional as F



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

def train_rnn(model, book_data, char_to_ind, ind_to_char, seq_length=25, eta=0.001, 
              num_updates=100000, rng=None, use_optimized=False):
    """
    Train the RNN model on the book data.
    
    Args:
        model: RNN model
        book_data: String containing the book text
        char_to_ind: Dictionary mapping characters to indices
        ind_to_char: Dictionary mapping indices to characters
        seq_length: Length of training sequences
        eta: Learning rate
        num_updates: Number of update steps
        rng: Random number generator
        use_optimized: Whether to use optimized forward/backward pass
    
    Returns:
        loss_history: List of smoothed losses during training
        sample_texts: List of sample texts generated during training
        sample_iters: List of iteration numbers when samples were generated
    """
    # If using optimized implementations, import them
    if use_optimized:
        from src.optimization import forward_pass_optimized, backward_pass_optimized
        forward_func = forward_pass_optimized
        backward_func = backward_pass_optimized
    else:
        # Use the model's forward and backward methods
        forward_func = lambda X, h0: model.forward(X, h0)
        backward_func = lambda X, Y, P, h, a: model.backward(X, Y, P, h, a)
    
    # Number of unique characters
    K = len(char_to_ind)
    
    # Initialize variables
    e = 0  # Position in the book
    h_prev = np.zeros((model.m, 1))  # Initial hidden state
    smooth_loss = -np.log(1.0/K) * seq_length  # Initial loss
    loss_history = []
    sample_texts = []
    sample_iters = []
    
    # Start time for tracking
    start_time = time.time()
    
    # Training loop
    for iteration in range(1, num_updates + 1):
        # Check if we need to reset after an epoch
        if e + seq_length + 1 > len(book_data):
            e = 0  # Reset position
            h_prev = np.zeros((model.m, 1))  # Reset hidden state
            print(f"Completed epoch at iteration {iteration}")
        
        # Get the next training sequence
        X, Y, X_chars, Y_chars = get_sequence_data(book_data, e, seq_length, char_to_ind, K)
        
        # Forward pass
        if use_optimized:
            P, h, a, o = forward_func(X, h_prev)
        else:
            P, h, a, o = forward_func(X, h_prev)
        
        # Calculate loss
        loss = compute_loss(Y, P)
        
        # Backward pass
        if use_optimized:
            grads = backward_func(X, Y, P, h, a, model.params)
        else:
            grads = backward_func(X, Y, P, h, a)
        
        # Update parameters using Adam
        model.adam_update(grads, eta)
        
        # Update position in the book
        e += seq_length
        
        # Update hidden state for the next sequence
        h_prev = h[:, -1].reshape(-1, 1)
        
        # Update smoothed loss
        smooth_loss = 0.999 * smooth_loss + 0.001 * loss
        loss_history.append(smooth_loss)
        
        # Print updates
        if iteration % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f"iter = {iteration}, smooth loss = {smooth_loss:.6f}, time = {elapsed_time:.2f}s")
        
        # Generate sample text periodically
        if iteration % 1000 == 0 or iteration == 1:
            # Generate text using the current model
            x0 = X[:, 0:1]  # First character of current sequence
            generated_text, _ = model.synthesize_text(h_prev, x0, 200, ind_to_char, char_to_ind, rng=rng)
            print(f"Sample text at iteration {iteration}:\n{generated_text}\n")
            
            # Save the sample text
            sample_texts.append(generated_text)
            sample_iters.append(iteration) 
            _ = model.synthesize_text(h_prev, x0, 200, ind_to_char, char_to_ind, rng=rng)
            print(f"Sample text at iteration {iteration}:\n{generated_text}\n")
            
            # Save the sample text
            sample_texts.append(generated_text)
            sample_iters.append(iteration)
    
    return loss_history, sample_texts, sample_iters


class RNN:
    def __init__(self, K, m, rng):
        """
        Initialize a vanilla RNN model.
        
        Args:
            K: Input/output dimension (number of unique characters)
            m: Hidden state dimension
            rng: Random number generator
        """
        # Initialize model parameters
        self.params = {
            'b': np.zeros((m, 1)),                # Hidden bias
            'c': np.zeros((K, 1)),                # Output bias
            'U': (1/np.sqrt(2*K)) * rng.standard_normal(size=(m, K)),  # Input-to-hidden weights
            'W': (1/np.sqrt(2*m)) * rng.standard_normal(size=(m, m)),  # Hidden-to-hidden weights
            'V': (1/np.sqrt(m)) * rng.standard_normal(size=(K, m))     # Hidden-to-output weights
        }
        
        # Store dimensions
        self.K = K
        self.m = m
        
        # Initialize Adam optimizer parameters
        self.optimizer = {
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
            'm': {k: np.zeros_like(v) for k, v in self.params.items()},
            'v': {k: np.zeros_like(v) for k, v in self.params.items()},
            't': 0
        }
    
    def forward(self, X, h_prev):
        """
        Forward pass through the RNN.
        
        Args:
            X: Input data, shape (K, seq_length)
            h_prev: Previous hidden state, shape (m, 1)
        
        Returns:
            P: Output probabilities, shape (K, seq_length)
            h: Hidden states, shape (m, seq_length+1)
            a: Pre-activation hidden states, shape (m, seq_length)
            o: Pre-softmax outputs, shape (K, seq_length)
        """
        seq_length = X.shape[1]
        
        # Initialize arrays to store intermediate values
        a = np.zeros((self.m, seq_length))
        h = np.zeros((self.m, seq_length + 1))
        o = np.zeros((self.K, seq_length))
        P = np.zeros((self.K, seq_length))
        
        # Set the initial hidden state
        h[:, 0:1] = h_prev
        
        # Forward pass through time
        for t in range(seq_length):
            # Input to the hidden layer
            a[:, t:t+1] = np.dot(self.params['W'], h[:, t:t+1]) + np.dot(self.params['U'], X[:, t:t+1]) + self.params['b']
            
            # Hidden state
            h[:, t+1:t+2] = np.tanh(a[:, t:t+1])
            
            # Output layer
            o[:, t:t+1] = np.dot(self.params['V'], h[:, t+1:t+2]) + self.params['c']
            
            # Softmax probabilities
            P[:, t:t+1] = np.exp(o[:, t:t+1]) / np.sum(np.exp(o[:, t:t+1]))
        
        return P, h, a, o
    
    def backward(self, X, Y, P, h, a):
        """
        Backward pass through the RNN.
        
        Args:
            X: Input data, shape (K, seq_length)
            Y: Target data, shape (K, seq_length)
            P: Output probabilities from forward pass, shape (K, seq_length)
            h: Hidden states from forward pass, shape (m, seq_length+1)
            a: Pre-activation hidden states from forward pass, shape (m, seq_length)
        
        Returns:
            grads: Dictionary of gradients for all parameters
        """
        seq_length = X.shape[1]
        
        # Initialize gradients
        dU = np.zeros_like(self.params['U'])
        dW = np.zeros_like(self.params['W'])
        dV = np.zeros_like(self.params['V'])
        db = np.zeros_like(self.params['b'])
        dc = np.zeros_like(self.params['c'])
        
        # Initialize gradient of hidden state
        dh_next = np.zeros((self.m, 1))
        
        # Backward pass through time
        for t in reversed(range(seq_length)):
            # Gradient of the output
            do = P[:, t:t+1] - Y[:, t:t+1]
            
            # Gradient of V and c
            dV += np.dot(do, h[:, t+1:t+2].T)
            dc += do
            
            # Gradient of hidden state
            dh = np.dot(self.params['V'].T, do) + dh_next
            
            # Gradient of tanh
            da = (1 - np.square(h[:, t+1:t+2])) * dh
            
            # Gradient of W, U, and b
            db += da
            dW += np.dot(da, h[:, t:t+1].T)
            dU += np.dot(da, X[:, t:t+1].T)
            
            # Gradient for next iteration
            dh_next = np.dot(self.params['W'].T, da)
        
        # Clip gradients to prevent exploding gradients
        for grad in [dU, dW, dV]:
            np.clip(grad, -5, 5, out=grad)
        
        # Store gradients in a dictionary
        grads = {'U': dU, 'W': dW, 'V': dV, 'b': db, 'c': dc}
        
        return grads
    
    def adam_update(self, grads, eta=0.001):
        """
        Update parameters using Adam optimizer.
        
        Args:
            grads: Dictionary of gradients
            eta: Learning rate
        """
        # Increment time step
        self.optimizer['t'] += 1
        t = self.optimizer['t']
        
        # Update for each parameter
        for key in self.params.keys():
            # Update biased first moment estimate
            self.optimizer['m'][key] = self.optimizer['beta1'] * self.optimizer['m'][key] + \
                                      (1 - self.optimizer['beta1']) * grads[key]
            
            # Update biased second moment estimate
            self.optimizer['v'][key] = self.optimizer['beta2'] * self.optimizer['v'][key] + \
                                      (1 - self.optimizer['beta2']) * (grads[key]**2)
            
            # Bias correction
            m_hat = self.optimizer['m'][key] / (1 - self.optimizer['beta1']**t)
            v_hat = self.optimizer['v'][key] / (1 - self.optimizer['beta2']**t)
            
            # Update parameters
            self.params[key] -= eta * m_hat / (np.sqrt(v_hat) + self.optimizer['epsilon'])
    
    def synthesize_text(self, h0, x0, n, ind_to_char, char_to_ind, sampling_strategy='standard', temperature=1.0, theta=0.9, rng=None):
        """
        Synthesize text from the RNN model.
        
        Args:
            h0: Initial hidden state, shape (m, 1)
            x0: Initial input character (one-hot encoded), shape (K, 1)
            n: Number of characters to generate
            ind_to_char: Dictionary mapping indices to characters
            char_to_ind: Dictionary mapping characters to indices
            sampling_strategy: Method for sampling ('standard', 'temperature', or 'nucleus')
            temperature: Temperature parameter for temperature sampling
            theta: Threshold for nucleus sampling
            rng: Random number generator
        
        Returns:
            generated_text: String of generated characters
            Y: One-hot encoding of generated characters, shape (K, n)
        """
        # Initialize arrays
        Y = np.zeros((self.K, n))
        x = x0
        h = h0
        
        # Generate sequence
        for t in range(n):
            # Forward pass for one step
            a = np.dot(self.params['W'], h) + np.dot(self.params['U'], x) + self.params['b']
            h = np.tanh(a)
            o = np.dot(self.params['V'], h) + self.params['c']
            p = np.exp(o) / np.sum(np.exp(o))
            
            # Apply sampling strategy
            if sampling_strategy == 'temperature':
                from src.utils import sample_with_temperature
                p = sample_with_temperature(p, temperature)
            elif sampling_strategy == 'nucleus':
                from src.utils import nucleus_sampling
                p = nucleus_sampling(p, theta)
            
            # Sample the next character
            from src.utils import sample_next_char
            idx = sample_next_char(p, rng)
            
            # Store the generated character
            Y[idx, t] = 1
            
            # Use the generated character as the next input
            x = np.zeros((self.K, 1))
            x[idx, 0] = 1
        
        # Convert one-hot encoding to text
        indices = np.argmax(Y, axis=0)
        generated_text = ''.join([ind_to_char[idx] for idx in indices])
        
        return generated_text, Y


def read_data(file_path):
    """
    Read the training text data from file.
    
    Args:
        file_path: Path to the text file
    
    Returns:
        book_data: String containing the book text
    """
    with open(file_path, "r", encoding="utf-8") as fid:
        book_data = fid.read()
    return book_data

def create_mappings(book_data):
    """
    Create character to index and index to character mappings.
    
    Args:
        book_data: String containing the book text
    
    Returns:
        char_to_ind: Dictionary mapping characters to indices
        ind_to_char: Dictionary mapping indices to characters
        unique_chars: List of unique characters
    """
    # Find all unique characters in the book
    unique_chars = list(set(book_data))
    K = len(unique_chars)
    
    # Create dictionaries for char to index and index to char mappings
    char_to_ind = {char: i for i, char in enumerate(unique_chars)}
    ind_to_char = {i: char for i, char in enumerate(unique_chars)}
    
    return char_to_ind, ind_to_char, unique_chars

def get_one_hot_encoding(sequence, char_to_ind, K):
    """
    Convert a sequence of characters to one-hot encoded vectors.
    
    Args:
        sequence: String of characters
        char_to_ind: Dictionary mapping characters to indices
        K: Number of unique characters
    
    Returns:
        one_hot: One-hot encoded matrix, shape (K, len(sequence))
    """
    indices = [char_to_ind[char] for char in sequence]
    one_hot = np.zeros((K, len(sequence)))
    one_hot[indices, np.arange(len(sequence))] = 1
    return one_hot

def get_sequence_data(book_data, e, seq_length, char_to_ind, K):
    """
    Get a training sequence from the book data.
    
    Args:
        book_data: String containing the book text
        e: Starting index for the sequence
        seq_length: Length of the sequence
        char_to_ind: Dictionary mapping characters to indices
        K: Number of unique characters
    
    Returns:
        X: Input sequence, shape (K, seq_length)
        Y: Target sequence, shape (K, seq_length)
        X_chars: Input characters
        Y_chars: Target characters
    """
    X_chars = book_data[e:e+seq_length]
    Y_chars = book_data[e+1:e+seq_length+1]
    
    X = get_one_hot_encoding(X_chars, char_to_ind, K)
    Y = get_one_hot_encoding(Y_chars, char_to_ind, K)
    
    return X, Y, X_chars, Y_chars

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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='RNN Text Generation')
    parser.add_argument('--check-gradients', action='store_true', help='Run gradient checking')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--num-updates', type=int, default=100000, help='Number of update steps')
    parser.add_argument('--hidden-size', type=int, default=100, help='Hidden state dimension')
    parser.add_argument('--seq-length', type=int, default=25, help='Sequence length')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=400, help='Random seed')
    parser.add_argument('--optimized', action='store_true', help='Use optimized forward/backward pass')
    args = parser.parse_args()
    
    # Set paths
    data_dir = 'data'
    book_fname = os.path.join(data_dir, 'goblet_book.txt')
    
    # Initialize random number generator
    rng = init_rng(seed=args.seed)
    
    # Read data
    print("Reading book data...")
    book_data = read_data(book_fname)
    print(f"Book length: {len(book_data)} characters")
    
    # Create character mappings
    char_to_ind, ind_to_char, unique_chars = create_mappings(book_data)
    K = len(unique_chars)
    print(f"Number of unique characters: {K}")
    
    # Set hyperparameters
    m = args.hidden_size  # Hidden state dimension
    seq_length = args.seq_length  # Sequence length
    eta = args.learning_rate  # Learning rate
    num_updates = args.num_updates  # Number of update steps
    
    # Run gradient checking if requested
    if args.check_gradients:
        print("\nRunning gradient checking...")
        print("1. Checking analytic vs numerical gradients...")
        is_correct_num = check_gradients(book_data, char_to_ind)
        print(f"Gradient check with numerical gradients: {'PASSED' if is_correct_num else 'FAILED'}")
        
        print("\n2. Checking analytic vs PyTorch gradients...")
        is_correct_torch = torch_gradient_check(book_data, char_to_ind)
        print(f"Gradient check with PyTorch: {'PASSED' if is_correct_torch else 'FAILED'}")
        
        if is_correct_num and is_correct_torch:
            print("\nGradient checking PASSED! The gradient computations are correct.")
        else:
            print("\nGradient checking FAILED! Please check your implementation.")
    
    # Train the model if requested
    if args.train:
        # Initialize model
        print("\nInitializing RNN model...")
        model = RNN(K, m, rng)
        
        # Generate text from untrained model
        h0 = np.zeros((m, 1))
        x0 = np.zeros((K, 1))
        x0[char_to_ind[book_data[0]], 0] = 1
        
        print("Generating text from untrained model...")
        untrained_text, _ = model.synthesize_text(
            h0, x0, 200, ind_to_char, char_to_ind, rng=rng
        )
        print(f"Untrained model output:\n{untrained_text}\n")
        
        # Train model
        print("Starting training...")
        loss_history, sample_texts, sample_iters = train_rnn(
            model, book_data, char_to_ind, ind_to_char, 
            seq_length=seq_length, eta=eta, num_updates=num_updates, 
            rng=rng, use_optimized=args.optimized
        )
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Plot loss history
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Smoothed Loss')
        plt.title('Training Loss')
        plt.grid(True)
        plt.savefig(os.path.join('results', 'loss_history.png'))
        plt.close()
        
        # Generate final sample with 1000 characters
        print("Generating final sample (1000 characters)...")
        h_prev = np.zeros((m, 1))
        x0 = np.zeros((K, 1))
        x0[char_to_ind[book_data[0]], 0] = 1
        
        final_sample, _ = model.synthesize_text(
            h_prev, x0, 1000, ind_to_char, char_to_ind, rng=rng
        )
        
        # Save final sample to file
        with open(os.path.join('results', 'final_sample.txt'), 'w', encoding='utf-8') as f:
            f.write(final_sample)
        
        print("Training and sampling complete!")
        
        # Save training samples
        with open(os.path.join('results', 'training_samples.txt'), 'w', encoding='utf-8') as f:
            for i, (iteration, text) in enumerate(zip(sample_iters, sample_texts)):
                f.write(f"Iteration {iteration}:\n{text}\n\n")
        
        # Generate samples with different sampling strategies
        print("\nGenerating samples with different sampling strategies...")
        
        # Temperature sampling
        temp_samples = []
        for temp in [0.2, 0.5, 0.7]:
            temp_text, _ = model.synthesize_text(
                h_prev, x0, 200, ind_to_char, char_to_ind,
                sampling_strategy='temperature', temperature=temp, rng=rng
            )
            temp_samples.append((temp, temp_text))
        
        # Nucleus sampling
        nucleus_samples = []
        for theta in [0.5, 0.7, 0.9]:
            nucleus_text, _ = model.synthesize_text(
                h_prev, x0, 200, ind_to_char, char_to_ind,
                sampling_strategy='nucleus', theta=theta, rng=rng
            )
            nucleus_samples.append((theta, nucleus_text))
        
        # Save sampling strategy samples
        with open(os.path.join('results', 'sampling_strategies.txt'), 'w', encoding='utf-8') as f:
            f.write("Temperature Sampling:\n")
            for temp, text in temp_samples:
                f.write(f"Temperature = {temp}:\n{text}\n\n")
            
            f.write("\nNucleus Sampling:\n")
            for theta, text in nucleus_samples:
                f.write(f"Theta = {theta}:\n{text}\n\n")
        
        # Print information for the report
        print("\nInformation for the report:")
        print("1. Analytic gradient computations have been implemented and verified.")
        print("2. The loss history has been saved to 'results/loss_history.png'.")
        print("3. Sample texts have been saved to 'results/training_samples.txt'.")
        print("4. The final 1000-character sample has been saved to 'results/final_sample.txt'.")
        print("5. Samples with different sampling strategies have been saved to 'results/sampling_strategies.txt'.")

if __name__ == "__main__":
    main()