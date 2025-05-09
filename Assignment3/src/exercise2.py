#!/usr/bin/env python
# coding: utf-8

import numpy as np
import argparse
def forward_pass(conv_outputs_mat, W1, W2, b1, b2, n_p, nf, n):
    """
    Perform forward pass of the network
    """
    # ReLU activation on flattened convolution outputs
    conv_flat = np.fmax(conv_outputs_mat.reshape((n_p * nf, n), order='C'), 0)

    # First layer
    s1 = W1 @ conv_flat + b1  # shape: (nh × 1)
    h = np.maximum(0, s1)     # ReLU activation

    # Second layer (output layer)
    s2 = W2 @ h + b2         # shape: (10 × n)

    # Softmax activation
    exp_scores = np.exp(s2)
    P = exp_scores / np.sum(exp_scores, axis=0)  # shape: (10 × n)

    return {
        'conv_flat': conv_flat,
        'h': h,
        'P': P
    }

def backward_pass(Y, P, h, conv_flat, W1, W2, conv_outputs_mat, n_p, nf, n, MX):
    """
    Perform backward pass to compute gradients according to the backpropagation equations
    from the assignment description.
    """
    # Compute gradients for the second layer (output layer)
    # This corresponds to the gradient of cross-entropy loss w.r.t. the softmax output
    # The gradient is -(Y - P) as per standard cross-entropy derivative
    # See equation (24) for label smoothing version: -(y_smooth - p)
    G_batch = -(Y - P)  # shape: (10 × n)

    # Apply batch normalization as per equation (20) and (21) where we divide by batch size |B|
    # L(B, Θ) = (1/|B|) * sum of losses over the batch
    grad_W2 = (G_batch @ h.T) / n  # Division by batch size n implements the (1/|B|) term
    grad_b2 = np.sum(G_batch, axis=1, keepdims=True) / n  # Sum over batch, then normalize

    # Compute gradients for the first layer (hidden layer)
    # Backpropagate the gradient through W2
    G_batch = W2.T @ G_batch
    # Apply ReLU gradient (derivative is 1 for positive values, 0 otherwise)
    # This corresponds to the derivative of equation (3): max(0, W_1 h + b_1)
    G_batch = G_batch * (h > 0)  # ReLU gradient
    grad_W1 = (G_batch @ conv_flat.T) / n  # Normalize by batch size
    grad_b1 = np.sum(G_batch, axis=1, keepdims=True) / n  # Normalize by batch size

    # Compute gradients for the filters (convolution layer)
    # Backpropagate the gradient through W1
    G_batch = W1.T @ G_batch
    # Apply ReLU gradient for the convolution outputs
    # This corresponds to the derivative of equation (1): max(0, X * F_i)
    G_batch = G_batch * (conv_flat > 0)  # ReLU gradient
    # Reshape to match the expected dimensions for einsum
    # This matches the shape of G in equation (19): G = [g_1, g_2, g_3]
    GG = G_batch.reshape((n_p, nf, n), order='C')

    # Compute gradients for the filters using Einstein summation
    # Transpose MX to prepare for the operation in equation (18): ∂L/∂F_all = MX^T * G
    MXt = np.transpose(MX, (1, 0, 2))

    # This implements equation (22): F^grad_all = (1/n) * sum_i=1^n [M(:,:,i)^T * G(:,:,i)]
    # The einsum operation performs the matrix multiplication for each sample in the batch
    # and the division by n implements the (1/|B|) term from equation (21)
    grad_Fs_flat = np.einsum('ijn,jln->il', MXt, GG, optimize=True) / n

    return {
        'grad_W2': grad_W2,
        'grad_b2': grad_b2,
        'grad_W1': grad_W1,
        'grad_b1': grad_b1,
        'grad_Fs_flat': grad_Fs_flat
    }

def compute_loss(Y, P):
    """
    Compute the cross-entropy loss
    """
    return -np.sum(Y * np.log(P)) / Y.shape[1]

def verify_gradients(computed_grads, load_data):
    """
    Verify computed gradients against provided data
    """
    print("Verifying grad_Fs_flat...")
    diff = np.abs(computed_grads['grad_Fs_flat'] - load_data['grad_Fs_flat'])
    print(f"Maximum absolute difference: {np.max(diff)}")
    print(f"Average absolute difference: {np.mean(diff)}")
    return diff

def detailed_gradient_verification(computed_grads, load_data):
    """
    Perform detailed verification of computed gradients
    """
    # Check shapes first
    print("Checking shapes...")
    print(f"Computed grad_Fs_flat shape: {computed_grads['grad_Fs_flat'].shape}")
    print(f"Expected grad_Fs_flat shape: {load_data['grad_Fs_flat'].shape}")

    # Compute element-wise differences
    diff = computed_grads['grad_Fs_flat'] - load_data['grad_Fs_flat']
    abs_diff = np.abs(diff)

    print("\nGradient difference statistics:")
    print(f"Maximum absolute difference: {np.max(abs_diff):.10f}")
    print(f"Average absolute difference: {np.mean(abs_diff):.10f}")
    print(f"Median absolute difference: {np.median(abs_diff):.10f}")
    print(f"Standard deviation of differences: {np.std(abs_diff):.10f}")

    # Check for numerical instability
    print("\nChecking for potential numerical issues...")
    print(f"Range of computed gradients: [{np.min(computed_grads['grad_Fs_flat']):.4f}, {np.max(computed_grads['grad_Fs_flat']):.4f}]")
    print(f"Range of expected gradients: [{np.min(load_data['grad_Fs_flat']):.4f}, {np.max(load_data['grad_Fs_flat']):.4f}]")

    # Find locations of largest differences
    worst_indices = np.argsort(abs_diff.flatten())[-5:][::-1]
    print("\nTop 5 worst differences:")
    for idx in worst_indices:
        i, j = np.unravel_index(idx, abs_diff.shape)
        print(f"Position ({i},{j}):")
        print(f"  Computed value: {computed_grads['grad_Fs_flat'][i,j]:.10f}")
        print(f"  Expected value: {load_data['grad_Fs_flat'][i,j]:.10f}")
        print(f"  Absolute difference: {abs_diff[i,j]:.10f}")

def verify_implementation(data, forward_results, backward_results):
    """
    Verify the implementation by checking intermediate values
    """
    print("Verifying forward pass intermediates...")
    if 'conv_flat' in data:
        conv_flat_diff = np.abs(forward_results['conv_flat'] - data['conv_flat'])
        print(f"conv_flat max difference: {np.max(conv_flat_diff):.10f}")

    if 'P' in data:
        P_diff = np.abs(forward_results['P'] - data['P'])
        print(f"P max difference: {np.max(P_diff):.10f}")

    return detailed_gradient_verification(backward_results, data)

def load_debug_data(data_path='../data/debug_info.npz'):
    """
    Load the necessary data for Exercise 2

    Args:
        data_path: Path to the data file (default: 'Assignment3_data.npz')

    Returns:
        dict: Dictionary containing all necessary data and parameters
    """
    # Load the data file
    load_data = np.load(data_path, allow_pickle=True)

    # Extract required parameters and data
    data = {
        # Network parameters
        'W1': load_data['W1'],          # shape: nh × (n_p * nf)
        'W2': load_data['W2'],          # shape: 10 × nh
        'b1': load_data['b1'],          # shape: nh × 1
        'b2': load_data['b2'],          # shape: 10 × 1

        # Forward pass intermediates (for verification)
        'conv_flat': load_data['conv_flat'],
        'X1': load_data['X1'],
        'P': load_data['P'],

        # Target labels
        'Y': load_data['Y'],            # shape: 10 × n

        # Additional data needed for gradient computation
        'conv_outputs_mat': load_data['conv_outputs_mat'],
        'MX': load_data['MX'],

        # For gradient verification
        'grad_Fs_flat': load_data['grad_Fs_flat']
    }

    # Extract dimensions from the data
    data['nf'] = load_data['nf'].item()    # number of filters
    data['nh'] = load_data['nh'].item()    # number of hidden units

    # Manually calculate n_p (number of pixels)
    # We can determine this from W1's shape: nh × (n_p * nf)
    # First dimension of W1 should be nh, second should be (n_p * nf)
    # So n_p = W1.shape[1] / nf
    data['n_p'] = load_data['W1'].shape[1] // data['nf']

    # Calculate batch size n from Y's shape (10 × n)
    data['n'] = load_data['Y'].shape[1]

    return data


def run_exercise_2(args):
    """
    Run Exercise 2: Compute gradients and verify them.
    """
    print("Running Exercise 2: Gradient Computation")
    print("-" * 50)
    
    try:

        
        # Load debugging data
        data = load_exercise2_data(args.debug_file)
        
        # Run forward pass
        forward_results = forward_pass(
            data['conv_outputs_mat'],
            data['W1'],
            data['W2'],
            data['b1'],
            data['b2'],
            data['n_p'],
            data['nf'],
            data['n']
        )
        
        # Compute loss
        loss = compute_loss(data['Y'], forward_results['P'])
        print(f"Cross-entropy loss: {loss:.4f}")
        
        # Verify forward pass
        print("Verifying forward pass intermediates...")
        if 'conv_flat' in data:
            conv_flat_diff = np.abs(forward_results['conv_flat'] - data['conv_flat'])
            print(f"conv_flat max difference: {np.max(conv_flat_diff):.10f}")
        
        if 'P' in data:
            P_diff = np.abs(forward_results['P'] - data['P'])
            print(f"P max difference: {np.max(P_diff):.10f}")
        
        # Run backward pass
        backward_results = backward_pass(
            data['Y'],
            forward_results['P'],
            forward_results['h'],
            forward_results['conv_flat'],
            data['W1'],
            data['W2'],
            data['conv_outputs_mat'],
            data['n_p'],
            data['nf'],
            data['n'],
            data['MX']
        )
        
        # Verify the gradients
        diff = verify_gradients(backward_results, data)
        
        # If needed, run detailed verification
        if np.max(diff) > 1e-6:
            print("\nDetailed gradient verification:")
            print("Checking shapes...")
            print(f"Computed grad_Fs_flat shape: {backward_results['grad_Fs_flat'].shape}")
            print(f"Expected grad_Fs_flat shape: {data['grad_Fs_flat'].shape}")
            
            # Further diagnostics
            print("\nChecking for potential numerical issues...")
            print(f"Range of computed gradients: [{np.min(backward_results['grad_Fs_flat']):.4f}, {np.max(backward_results['grad_Fs_flat']):.4f}]")
            print(f"Range of expected gradients: [{np.min(data['grad_Fs_flat']):.4f}, {np.max(data['grad_Fs_flat']):.4f}]")
        else:
            print("Gradients verified successfully!")
        
        print("\nExercise 2 completed successfully!")
        
    except Exception as e:
        print(f"Error in Exercise 2: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Exercise 2: Gradient Computation')
    parser.add_argument('--debug_file', type=str, default='data/debug_info.npz',
                        help='Path to the debug data file')
    args = parser.parse_args()
    run_exercise_2(args)

if __name__ == "__main__":
    main()