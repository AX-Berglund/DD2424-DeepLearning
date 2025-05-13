import numpy as np
import time
from src.utils import compute_loss
from src.data import get_sequence_data

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