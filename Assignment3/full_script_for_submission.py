
#!/usr/bin/env python
# coding: utf-8

import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import os
import sys
import pickle

from datetime import datetime
from tqdm import tqdm






def prepare_data(X, Fs):
    """
    Prepare data for convolution operations.
    
    Parameters:
        X: Input data of shape (3072, n) where n is the number of images
        Fs: Filters of shape (f, f, 3, nf) where f is filter size and nf is number of filters
        
    Returns:
        X_ims: Reshaped images of shape (32, 32, 3, n)
        MX: Matrix of patches for matrix multiplication of shape (n_p, f*f*3, n)
        Fs_flat: Flattened filters of shape (f*f*3, nf)
        dimensions: Dictionary with key dimensions (f, nf, n, n_p)
    """
    # Get dimensions
    f = Fs.shape[0]  # filter size
    nf = Fs.shape[3]  # number of filters
    n = X.shape[1]  # number of images
    n_p = (32 // f) ** 2  # number of patches per image
    
    # Reshape and transpose X to get images in the right format
    X_ims = np.transpose(X.reshape((32, 32, 3, n), order='F'), (1, 0, 2, 3))
    
    # Initialize MX matrix for matrix multiplication and einsum methods
    MX = np.zeros((n_p, f * f * 3, n))
    
    # Fill MX with patches
    for i in range(n):
        patch_idx = 0
        for h in range(0, 32, f):
            for w in range(0, 32, f):
                # Extract patch and reshape it to a row vector
                patch = X_ims[h:h+f, w:w+f, :, i]
                MX[patch_idx, :, i] = patch.reshape((1, f * f * 3), order='C')
                patch_idx += 1
    
    # Flatten the filters
    Fs_flat = Fs.reshape((f * f * 3, nf), order='C')
    
    dimensions = {
        'f': f,
        'nf': nf,
        'n': n,
        'n_p': n_p
    }
    
    return X_ims, MX, Fs_flat, dimensions

def conv_dot_product(X_ims, Fs, dimensions):
    """
    Compute convolution using direct dot product method.
    
    Parameters:
        X_ims: Reshaped images of shape (32, 32, 3, n)
        Fs: Filters of shape (f, f, 3, nf)
        dimensions: Dictionary with dimensions
        
    Returns:
        conv_outputs: Convolution outputs of shape (n_p, nf, n)
    """
    f = dimensions['f']
    nf = dimensions['nf']
    n = dimensions['n']
    n_p = dimensions['n_p']
    
    conv_outputs = np.zeros((n_p, nf, n))
    
    # Loop through each image
    for i in range(n):
        # Loop through each filter
        for j in range(nf):
            patch_idx = 0
            # Loop through patches in the image
            for h in range(0, 32, f):
                for w in range(0, 32, f):
                    # Extract patch
                    patch = X_ims[h:h+f, w:w+f, :, i]
                    # Get filter
                    filter = Fs[:, :, :, j]
                    # Compute dot product
                    result = np.multiply(patch, filter).sum()
                    conv_outputs[patch_idx, j, i] = result
                    patch_idx += 1
                    
    return conv_outputs

def conv_matrix_mult(MX, Fs_flat, dimensions):
    """
    Compute convolution using matrix multiplication method.
    
    Parameters:
        MX: Matrix of patches of shape (n_p, f*f*3, n)
        Fs_flat: Flattened filters of shape (f*f*3, nf)
        dimensions: Dictionary with dimensions
        
    Returns:
        conv_outputs: Convolution outputs of shape (n_p, nf, n)
    """
    nf = dimensions['nf']
    n = dimensions['n']
    n_p = dimensions['n_p']
    
    # Initialize output array for matrix multiplication version
    conv_outputs = np.zeros((n_p, nf, n))
    
    # Compute convolution using matrix multiplication
    for i in range(n):
        conv_outputs[:, :, i] = np.matmul(MX[:, :, i], Fs_flat)
        
    return conv_outputs

def conv_einsum(MX, Fs_flat, dimensions):
    """
    Compute convolution using Einstein summation method.
    
    Parameters:
        MX: Matrix of patches of shape (n_p, f*f*3, n)
        Fs_flat: Flattened filters of shape (f*f*3, nf)
        dimensions: Dictionary with dimensions
        
    Returns:
        conv_outputs: Convolution outputs of shape (n_p, nf, n)
    """
    # Compute convolution using einsum
    conv_outputs = np.einsum('ijn,jl->iln', MX, Fs_flat, optimize=True)
    
    return conv_outputs

def time_method(method, X_ims, MX, Fs, Fs_flat, dimensions, runs=5):
    """
    Time a specific convolution method.
    
    Parameters:
        method: String specifying the method ('dot', 'matmul', or 'einsum')
        X_ims: Reshaped images of shape (32, 32, 3, n)
        MX: Matrix of patches of shape (n_p, f*f*3, n)
        Fs: Filters of shape (f, f, 3, nf)
        Fs_flat: Flattened filters of shape (f*f*3, nf)
        dimensions: Dictionary with dimensions
        runs: Number of runs for timing
        
    Returns:
        mean_time: Mean execution time
        std_time: Standard deviation of execution time
        outputs: Convolution outputs from the last run
    """
    times = []
    outputs = None
    
    for _ in range(runs):
        start = time.time()
        
        if method == "dot":
            outputs = conv_dot_product(X_ims, Fs, dimensions)
        elif method == "matmul":
            outputs = conv_matrix_mult(MX, Fs_flat, dimensions)
        else:  # einsum
            outputs = conv_einsum(MX, Fs_flat, dimensions)
            
        times.append(time.time() - start)
        
    return np.mean(times), np.std(times), outputs

def verify_outputs(outputs, expected_outputs, method, dimensions):
    """
    Verify the outputs against expected outputs.
    
    Parameters:
        outputs: Computed convolution outputs
        expected_outputs: Expected outputs for verification
        method: Method used for computation
        dimensions: Dictionary with dimensions
        
    Returns:
        max_diff: Maximum absolute difference between outputs and expected outputs
    """
    nf = dimensions['nf']
    n = dimensions['n']
    
    if method == "dot":
        # Reshape for comparison with expected outputs
        outputs_reshaped = outputs.reshape((8, 8, nf, n))
        max_diff = np.abs(outputs_reshaped - expected_outputs).max()
    else:
        # For matmul and einsum, reshape expected outputs to match our outputs
        expected_reshaped = expected_outputs.reshape((dimensions['n_p'], nf, n))
        max_diff = np.abs(outputs - expected_reshaped).max()
        
    return max_diff

def load_exercise1_data(file_path):
    """
    Load debug data from NPZ file.
    
    Parameters:
        file_path: Path to the debug data file
        
    Returns:
        X: Input data
        Fs: Filters
        conv_outputs: Expected convolution outputs
    """
    load_data = np.load(file_path)
    X = load_data['X']
    Fs = load_data['Fs']
    conv_outputs = load_data['conv_outputs']
    
    return X, Fs, conv_outputs



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

def load_exercise2_data(data_path='../data/debug_info.npz'):
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

# ConvolutionalNetwork class - optimized version
class ConvolutionalNetwork:
    def __init__(self, f=4, nf=10, nh=50):
        """
        Initialize a convolutional network with a patchify layer.

        Args:
            f: Filter/patch size (2, 4, 8, or 16)
            nf: Number of filters
            nh: Number of hidden units
        """
        self.f = f  # Filter size (also stride)
        self.nf = nf  # Number of filters
        self.nh = nh  # Number of hidden units

        # Calculate number of patches per image
        self.n_p = (32 // f) ** 2

        # Input dimensions
        self.input_dim = 3072  # 32x32x3
        self.output_dim = 10  # 10 classes for CIFAR-10

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with Xavier initialization."""
        # Initialize filters (f x f x 3 x nf)
        filter_size = self.f * self.f * 3
        self.Fs = np.random.normal(0, 1/np.sqrt(filter_size), (self.f, self.f, 3, self.nf))
        
        # Flatten filters for easier computation (f*f*3, nf)
        self.Fs_flat = self.Fs.reshape((self.f * self.f * 3, self.nf), order='C')
        
        # Hidden layer weights
        self.W1 = np.random.normal(0, 1/np.sqrt(self.n_p * self.nf), (self.nh, self.n_p * self.nf))
        self.b1 = np.zeros((self.nh, 1))
        
        # Output layer weights
        self.W2 = np.random.normal(0, 1/np.sqrt(self.nh), (10, self.nh))
        self.b2 = np.zeros((10, 1))
        
        # Precompute patch indices for faster forward pass
        self._precompute_patch_indices()
        
    def _precompute_patch_indices(self):
        """Precompute patch indices to avoid recalculating them in each forward pass"""
        self.patch_indices = []
        for i in range(0, 32, self.f):
            for j in range(0, 32, self.f):
                # Calculate the indices for this patch
                indices = []
                for ii in range(self.f):
                    for jj in range(self.f):
                        for c in range(3):  # RGB channels
                            pixel_idx = c * 32 * 32 + (i + ii) * 32 + (j + jj)
                            indices.append(pixel_idx)
                self.patch_indices.append(indices)

    def forward(self, X):
        """
        Forward pass through the network - optimized.

        Args:
            X: Input data (3072, batch_size)

        Returns:
            P: Probability outputs (10, batch_size)
            cache: Cached values for backward pass
        """
        batch_size = X.shape[1]
        
        # OPTIMIZATION: Direct reshaping without multiple transpose operations
        # Create MX matrix for convolution directly from X using precomputed indices
        MX = np.zeros((self.n_p, self.f * self.f * 3, batch_size), dtype=X.dtype)
        
        # Fill MX with image patches using vectorized operations
        for idx, patch_indices in enumerate(self.patch_indices):
            MX[idx, :, :] = X[patch_indices, :]
        
        # Apply convolution using optimized einsum
        # 'ijn,jk->ikn' means:
        # i: patch index (n_p)
        # j: filter dimension (f*f*3)
        # k: filter index (nf)
        # n: batch dimension
        conv_outputs = np.einsum('ijn,jk->ikn', MX, self.Fs_flat, optimize='optimal')
        
        # Apply ReLU to convolution outputs
        conv_relu = np.maximum(0, conv_outputs)
        
        # Reshape to (n_p*nf, batch_size) for the dense layer
        conv_flat = conv_relu.reshape((self.n_p * self.nf, batch_size), order='C')
        
        # Forward through hidden layer - use optimized matrix multiplication
        s1 = self.W1 @ conv_flat + self.b1  # Using @ operator for matrix multiplication
        h1 = np.maximum(0, s1)  # ReLU activation
        
        # Forward through output layer
        s = self.W2 @ h1 + self.b2  # Using @ operator
        
        # Softmax activation - optimized for numerical stability
        # Subtract max for numerical stability
        s_shifted = s - np.max(s, axis=0, keepdims=True)
        exp_s = np.exp(s_shifted)
        P = exp_s / np.sum(exp_s, axis=0, keepdims=True)
        
        # Cache values for backward pass
        cache = {
            'X': X,
            'MX': MX,
            'conv_outputs': conv_outputs,
            'conv_relu': conv_relu,
            'conv_flat': conv_flat,
            's1': s1,
            'h1': h1,
            's': s
        }
        
        return P, cache

    def backward(self, Y, P, lambda_reg=0.0, cache=None):
        """
        Backward pass to compute gradients - optimized.

        Args:
            Y: One-hot encoded labels (10, batch_size)
            P: Predicted probabilities (10, batch_size)
            lambda_reg: L2 regularization parameter
            cache: Values cached during forward pass

        Returns:
            grads: Dictionary with gradients for all parameters
        """
        if cache is None:
            raise ValueError("Cache must be provided for backward pass")
        
        batch_size = Y.shape[1]
        
        # Gradient of cross-entropy loss with respect to softmax output
        G_batch = -(Y - P)  # (10, batch_size)
        
        # Gradient w.r.t W2, b2 - use matrix multiplication operator
        grad_W2 = (1/batch_size) * (G_batch @ cache['h1'].T) + 2 * lambda_reg * self.W2
        grad_b2 = (1/batch_size) * np.sum(G_batch, axis=1, keepdims=True)
        
        # Gradient w.r.t h1
        G_h1 = self.W2.T @ G_batch  # (nh, batch_size)
        
        # Gradient through ReLU
        G_s1 = G_h1 * (cache['s1'] > 0)  # (nh, batch_size)
        
        # Gradient w.r.t W1, b1
        grad_W1 = (1/batch_size) * (G_s1 @ cache['conv_flat'].T) + 2 * lambda_reg * self.W1
        grad_b1 = (1/batch_size) * np.sum(G_s1, axis=1, keepdims=True)
        
        # Gradient w.r.t conv_flat
        G_conv_flat = self.W1.T @ G_s1  # (n_p*nf, batch_size)
        
        # Reshape to match conv_relu
        G_conv_relu = G_conv_flat.reshape((self.n_p, self.nf, batch_size), order='C')
        
        # Gradient through ReLU
        G_conv = G_conv_relu * (cache['conv_outputs'] > 0)  # (n_p, nf, batch_size)
        
        # Use optimized einsum for gradient computation
        MXt = np.transpose(cache['MX'], (1, 0, 2))  # (f*f*3, n_p, batch_size)
        
        # OPTIMIZED EINSUM OPERATION
        grad_Fs_flat = np.einsum('ijn,jkn->ik', MXt, G_conv, optimize='optimal') / batch_size + 2 * lambda_reg * self.Fs_flat
        
        # Reshape grad_Fs back to original shape
        grad_Fs = grad_Fs_flat.reshape(self.Fs.shape, order='C')
        
        grads = {
            'W1': grad_W1,
            'b1': grad_b1,
            'W2': grad_W2,
            'b2': grad_b2,
            'Fs': grad_Fs,
            'Fs_flat': grad_Fs_flat
        }
        
        return grads

    def update_parameters(self, grads, learning_rate):
        """
        Update network parameters using gradient descent.

        Args:
            grads: Dictionary with gradients
            learning_rate: Learning rate for update
        """
        self.W1 -= learning_rate * grads['W1']
        self.b1 -= learning_rate * grads['b1']
        self.W2 -= learning_rate * grads['W2']
        self.b2 -= learning_rate * grads['b2']
        self.Fs -= learning_rate * grads['Fs']
        self.Fs_flat = self.Fs.reshape((self.f * self.f * 3, self.nf), order='C')

    def compute_loss_and_accuracy(self, X, Y, lambda_reg=0.0):
        """
        Compute loss and accuracy for dataset.

        Args:
            X: Input data (3072, n_samples)
            Y: One-hot encoded labels (10, n_samples)
            lambda_reg: L2 regularization parameter

        Returns:
            loss: Cross-entropy loss with L2 regularization
            accuracy: Classification accuracy
        """
        # Forward pass
        P, _ = self.forward(X)
        
        # Cross-entropy loss
        n_samples = X.shape[1]
        cross_entropy = -np.sum(Y * np.log(P + 1e-10)) / n_samples
        
        # L2 regularization
        l2_reg = lambda_reg * (np.sum(self.W1**2) + np.sum(self.W2**2) + np.sum(self.Fs**2))
        
        # Total loss
        loss = cross_entropy + l2_reg
        
        # Accuracy
        predictions = np.argmax(P, axis=0)
        targets = np.argmax(Y, axis=0)
        accuracy = np.mean(predictions == targets)
        
        return loss, accuracy


# Optimized training function
def train_with_cyclical_lr(model, X_train, Y_train, X_val, Y_val, CLRparams, lambda_reg=0.0, 
                           use_label_smoothing=False, epsilon=0.1, logging_freq=10, verbose=False):
    """
    Train the network using mini-batch gradient descent with cyclical learning rates.
    """
    # Extract parameters
    batch_size = CLRparams["n_batch"]
    eta_min = CLRparams["eta_min"]
    eta_max = CLRparams["eta_max"]
    initial_step_size = CLRparams["n_s"]
    n_cycles = CLRparams["n_cycles"]

    # Calculate step sizes for each cycle (doubling each time)
    step_sizes = [initial_step_size * (2 ** i) for i in range(n_cycles)]

    # Apply label smoothing if requested
    if use_label_smoothing:
        Y_train_smooth = apply_label_smoothing(Y_train, epsilon)
    else:
        Y_train_smooth = Y_train.copy()  # Use copy to avoid modifying original

    # Get data dimensions
    n_train = X_train.shape[1]

    # Calculate total updates
    total_updates = sum(2 * step_size for step_size in step_sizes)

    # Initialize history dictionary
    history = {
        "loss_train": [], "loss_val": [],
        "acc_train": [], "acc_val": [],
        "update_steps": [],
        "learning_rates": []
    }

    # Initialize tracking variables
    update_step = 0
    current_cycle = 0
    cycle_step = 0

    # Precompute validation data metrics for initial state
    val_loss, val_acc = model.compute_loss_and_accuracy(X_val, Y_val, lambda_reg)
    train_loss, train_acc = model.compute_loss_and_accuracy(X_train, Y_train, lambda_reg)
    
    # Add initial metrics to history
    history["loss_train"].append(train_loss)
    history["acc_train"].append(train_acc)
    history["loss_val"].append(val_loss)
    history["acc_val"].append(val_acc)
    history["update_steps"].append(0)
    history["learning_rates"].append(eta_min)

    # Start timer
    start_time = time.time()
    
    # OPTIMIZATION: Pre-allocate arrays for mini-batch indices
    all_indices = np.arange(n_train)
    batch_indices = []
    num_batches = n_train // batch_size
    for j in range(num_batches):
        j_start = j * batch_size
        j_end = min(j_start + batch_size, n_train)
        batch_indices.append((j_start, j_end))

    # Create progress bar for total updates
    pbar = tqdm(total=total_updates, desc="Training Progress", disable=not verbose)

    # Continue training until total updates reached
    while update_step < total_updates:
        # Shuffle data for each epoch
        shuffle_idx = np.random.permutation(n_train)
        X_shuffled = X_train[:, shuffle_idx]
        Y_shuffled = Y_train_smooth[:, shuffle_idx]

        # Process mini-batches
        for j_start, j_end in batch_indices:
            # Skip if we've done enough updates
            if update_step >= total_updates:
                break

            # Extract mini-batch
            X_batch = X_shuffled[:, j_start:j_end]
            Y_batch = Y_shuffled[:, j_start:j_end]

            # Forward pass
            P_batch, cache = model.forward(X_batch)

            # Backward pass
            grads = model.backward(Y_batch, P_batch, lambda_reg, cache)

            # Get current learning rate - precompute for efficiency
            eta = eta_min + ((eta_max - eta_min) * cycle_step / step_sizes[current_cycle]) if cycle_step < step_sizes[current_cycle] else \
                  eta_max - ((eta_max - eta_min) * (cycle_step - step_sizes[current_cycle]) / step_sizes[current_cycle])

            # Update model parameters
            model.update_parameters(grads, eta)

            # Update cycle tracking
            cycle_step += 1
            update_step += 1

            # Update progress bar
            pbar.update(1)

            # Check if cycle is complete
            if cycle_step >= 2 * step_sizes[current_cycle]:
                current_cycle += 1
                cycle_step = 0

                # Break if all cycles complete
                if current_cycle >= n_cycles:
                    break

            # Log metrics periodically - avoid expensive computations too often
            if update_step % logging_freq == 0:
                # Compute metrics
                train_loss, train_acc = model.compute_loss_and_accuracy(X_train, Y_train, lambda_reg)
                val_loss, val_acc = model.compute_loss_and_accuracy(X_val, Y_val, lambda_reg)
                
                # Store all metrics and learning rate
                history["learning_rates"].append(eta)
                history["update_steps"].append(update_step)
                history["loss_train"].append(train_loss)
                history["acc_train"].append(train_acc)
                history["loss_val"].append(val_loss)
                history["acc_val"].append(val_acc)
                
                # Update progress bar description with current metrics
                if verbose:
                    pbar.set_description(f"Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Close progress bar
    pbar.close()

    # Record training time
    training_time = time.time() - start_time
    history["training_time"] = training_time

    if verbose:
        print(f"\nTraining completed in {training_time:.2f} seconds")
        # Final metrics already stored in the last iteration
        print(f"Final validation accuracy: {history['acc_val'][-1]:.4f}")

    return history

def load_cifar_data(cifar10_path, n_train=49000, dtype=np.float32):
    """
    Load and prepare CIFAR-10 data properly.
    Combines multiple batches for training and reserves some for validation.
    
    Args:
        cifar10_path: Path to CIFAR-10 data directory
        n_train: Number of training samples to use
        dtype: Data type for arrays
    
    Returns:
        X_train, Y_train, X_val, Y_val, X_test, Y_test: Prepared datasets
    """
    # Load training batches 1-5
    X_train_full = []
    Y_train_full = []
    
    # Load all 5 training batches and combine them
    for batch_id in range(1, 6):
        X_batch, Y_batch, _ = load_batch(batch_id, cifar10_path, dtype)
        X_train_full.append(X_batch)
        Y_train_full.append(Y_batch)
    
    # Concatenate all batches
    X_train_full = np.concatenate(X_train_full, axis=1)  # Concatenate along samples dimension
    Y_train_full = np.concatenate(Y_train_full, axis=1)
    
    # Split into training and validation
    # Use first n_train samples for training
    X_train_raw = X_train_full[:, :n_train]
    Y_train = Y_train_full[:, :n_train]
    
    # Use remaining samples for validation
    n_val = X_train_full.shape[1] - n_train
    X_val_raw = X_train_full[:, n_train:]
    Y_val = Y_train_full[:, n_train:]
    
    # Load test set
    X_test_raw, Y_test, _ = load_batch("test_batch", cifar10_path, dtype)
    
    # Normalize data using training set statistics
    X_train, X_val, X_test = preprocess_data(X_train_raw, X_val_raw, X_test_raw)
    
    print(f"Data loaded: {X_train.shape[1]} training, {X_val.shape[1]} validation, {X_test.shape[1]} test samples")
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

# Optimized data loading function
def load_batch(batch_id, cifar10_path="../data/cifar-10-batches-py", dtype=np.float32):  # Changed to float32
    """
    Load a CIFAR-10 batch file and return image data, one-hot labels, and raw labels.
    """
    # Construct full file path
    if isinstance(batch_id, str):
        batch_file = os.path.join(cifar10_path, batch_id)
    else:   
        batch_file = os.path.join(cifar10_path, f"data_batch_{batch_id}")

    # Load the CIFAR-10 batch file
    with open(batch_file, 'rb') as file:
        batch = pickle.load(file, encoding='bytes')

    # Extract image data and labels
    images = batch[b'data']  # Shape (10000, 3072)
    labels = np.array(batch[b'labels'])  # Shape (10000,)

    # Convert image data to float and normalize to [0,1]
    X = images.astype(dtype) / 255.0  # Shape (10000, 3072)

    # Transpose X to match required shape (3072, 10000)
    X = X.T  # Shape (3072, 10000)

    # Convert labels to one-hot encoding using optimized approach
    K = 10  # Number of classes in CIFAR-10
    n_samples = X.shape[1]
    Y = np.zeros((K, n_samples), dtype=dtype)
    Y[labels, np.arange(n_samples)] = 1  # Assign 1s for correct labels

    # Make sure labels are (1, n_samples)
    labels = labels.reshape(1, len(labels))

    return X, Y, labels


def preprocess_data(X_train_raw, X_val_raw, X_test_raw):
    """
    Normalizes the dataset based on training set mean and standard deviation.
    """
    # Use float32 for better performance
    if X_train_raw.dtype != np.float32:
        X_train_raw = X_train_raw.astype(np.float32)
    if X_val_raw.dtype != np.float32:
        X_val_raw = X_val_raw.astype(np.float32)
    if X_test_raw.dtype != np.float32:
        X_test_raw = X_test_raw.astype(np.float32)
        
    X_train_mean = np.mean(X_train_raw, axis=1, keepdims=True)
    X_train_std = np.std(X_train_raw, axis=1, keepdims=True)

    X_train = (X_train_raw - X_train_mean) / X_train_std
    X_val = (X_val_raw - X_train_mean) / X_train_std
    X_test = (X_test_raw - X_train_mean) / X_train_std

    return X_train, X_val, X_test


def apply_label_smoothing(Y, epsilon=0.1):
    """
    Apply label smoothing to one-hot encoded labels.
    """
    K = Y.shape[0]  # Number of classes
    Y_smooth = (1 - epsilon) * Y + epsilon / K
    return Y_smooth


def save_training_results(args, history, test_acc, clr_params, lambda_reg, log_dir="results/exercise3"):
    """
    Save training results to files in the results directory.
    
    Args:
        history: Dictionary containing training history
        test_acc: Final test accuracy
        architecture: Architecture number
        log_dir: Directory to save results
    """
    # Create results directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Append summary to the training log file
    log_file = os.path.join(log_dir, f'architecture_{args.architecture}_train_log.txt')
    with open(log_file, 'a') as f:
        f.write("\n" + "="*50 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("="*50 + "\n")
        f.write(f"Training Summary for Architecture {args.architecture}\n")
        f.write(f"Training time: {history['training_time']:.2f} seconds\n")
        f.write(f"Final validation accuracy: {history['acc_val'][-1]:.4f}\n")
        f.write(f"Final test accuracy: {test_acc:.4f}\n")
        f.write("\nTraining Parameters:\n")
        f.write(f"Number of training samples: {args.n_train}\n")
        f.write(f"Number of threads: {args.num_threads}\n")
        f.write(f"Learning rate range: {clr_params['eta_min']} to {clr_params['eta_max']}\n")
        f.write(f"Number of cycles: {clr_params['n_cycles']}\n")
        f.write(f"Batch size: {clr_params['n_batch']}\n")
        f.write(f"Lambda regularization: {lambda_reg}\n")



def save_training_results(history, test_acc, architecture, args, clr_params, lambda_reg, log_dir="results/exercise4"):
    """
    Save training results to files in the results directory.
    
    Args:
        history: Dictionary containing training history
        test_acc: Final test accuracy
        architecture: Architecture number is 5
        args: Command line arguments containing n_train and num_threads
        clr_params: Dictionary containing cyclical learning rate parameters
        lambda_reg: Regularization parameter
        log_dir: Directory to save results
    """
    # Create results directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Append summary to the training log file
    log_file = os.path.join(log_dir, f'architecture_{architecture}_train_log.txt')
    with open(log_file, 'a') as f:
        f.write("\n" + "="*50 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("="*50 + "\n")
        f.write(f"Training Summary for Architecture {architecture}\n")
        f.write(f"Training time: {history['training_time']:.2f} seconds\n")
        f.write(f"Final validation accuracy: {history['acc_val'][-1]:.4f}\n")
        f.write(f"Final test accuracy: {test_acc:.4f}\n")
        f.write("\nTraining Parameters:\n")
        f.write(f"Number of training samples: {args.n_train}\n")
        f.write(f"Number of threads: {args.num_threads}\n")
        f.write(f"Learning rate range: {clr_params['eta_min']} to {clr_params['eta_max']}\n")
        f.write(f"Number of cycles: {clr_params['n_cycles']}\n")
        f.write(f"Batch size: {clr_params['n_batch']}\n")
        f.write(f"Lambda regularization: {lambda_reg}\n")


def train_longer_with_increasing_steps(X_train, Y_train, X_val, Y_val, X_test, Y_test, clr_params, lambda_reg):
    """
    Train a larger network with increasing step sizes and compare with label smoothing.
    
    Args:
        X_train, Y_train: Training data
        X_val, Y_val: Validation data
        X_test, Y_test: Test data
        clr_params: Cyclical learning rate parameters
        lambda_reg: L2 regularization parameter
    """
    # Architecture 5 parameters
    f = 4
    nf = 40
    nh = 300
    
    # Modify CLR parameters for longer training
    clr_params["n_cycles"] = 4
    clr_params["n_s"] = 800
    
    # Train without label smoothing
    print("\nTraining without label smoothing...")
    model_no_smooth = ConvolutionalNetwork(f=f, nf=nf, nh=nh)
    history_no_smooth = train_with_cyclical_lr(
        model_no_smooth, X_train, Y_train, X_val, Y_val,
        clr_params, lambda_reg,
        use_label_smoothing=False,
        logging_freq=200, verbose=True
    )
    
    # Compute final test accuracy without smoothing
    test_loss_no_smooth, test_acc_no_smooth = model_no_smooth.compute_loss_and_accuracy(X_test, Y_test, lambda_reg)
    
    # Save results without smoothing
    save_training_results(history_no_smooth, test_acc_no_smooth, 5, args, clr_params, lambda_reg)
    
    # Plot results without smoothing
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_no_smooth["update_steps"], history_no_smooth["loss_train"], label="Training Loss")
    plt.plot(history_no_smooth["update_steps"], history_no_smooth["loss_val"], label="Validation Loss")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.title("Without Label Smoothing - Loss Curves")
    plt.legend()
    plt.grid(True)
    
    # Train with label smoothing
    print("\nTraining with label smoothing...")
    model_smooth = ConvolutionalNetwork(f=f, nf=nf, nh=nh)
    history_smooth = train_with_cyclical_lr(
        model_smooth, X_train, Y_train, X_val, Y_val,
        clr_params, lambda_reg,
        use_label_smoothing=True,
        logging_freq=200, verbose=True
    )
    
    # Compute final test accuracy with smoothing
    test_loss_smooth, test_acc_smooth = model_smooth.compute_loss_and_accuracy(X_test, Y_test, lambda_reg)
    
    # Save results with smoothing
    save_training_results(history_smooth, test_acc_smooth, 5, args, clr_params, lambda_reg)
    
    # Plot results with smoothing
    plt.subplot(1, 2, 2)
    plt.plot(history_smooth["update_steps"], history_smooth["loss_train"], label="Training Loss")
    plt.plot(history_smooth["update_steps"], history_smooth["loss_val"], label="Validation Loss")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.title("With Label Smoothing - Loss Curves")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'results/exercise4/architecture_5_comparison_{timestamp}.png')
    plt.close()
    
    # Print comparison summary
    print("\nComparison Summary:")
    print(f"Without Label Smoothing - Test Accuracy: {test_acc_no_smooth:.4f}")
    print(f"With Label Smoothing - Test Accuracy: {test_acc_smooth:.4f}")
    
    return history_no_smooth, history_smooth

def compare_label_smoothing(X_train, Y_train, X_val, Y_val, X_test, Y_test, clr_params, lambda_reg):
    """
    Compare training with and without label smoothing for architecture 5.
    """
    # Architecture 5 parameters
    f = 4
    nf = 40
    nh = 300
    
    # Train without label smoothing
    print("\nTraining without label smoothing...")
    model_no_smooth = ConvolutionalNetwork(f=f, nf=nf, nh=nh)
    history_no_smooth = train_with_cyclical_lr(
        model_no_smooth, X_train, Y_train, X_val, Y_val,
        clr_params, lambda_reg,
        use_label_smoothing=False,
        logging_freq=200, verbose=True
    )
    
    # Train with label smoothing
    print("\nTraining with label smoothing...")
    model_smooth = ConvolutionalNetwork(f=f, nf=nf, nh=nh)
    history_smooth = train_with_cyclical_lr(
        model_smooth, X_train, Y_train, X_val, Y_val,
        clr_params, lambda_reg,
        use_label_smoothing=True,
        logging_freq=200, verbose=True
    )
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_no_smooth["update_steps"], history_no_smooth["loss_train"], label="Training Loss")
    plt.plot(history_no_smooth["update_steps"], history_no_smooth["loss_val"], label="Validation Loss")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.title("Without Label Smoothing")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history_smooth["update_steps"], history_smooth["loss_train"], label="Training Loss")
    plt.plot(history_smooth["update_steps"], history_smooth["loss_val"], label="Validation Loss")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.title("With Label Smoothing")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'results/exercise4/label_smoothing_comparison_{timestamp}.png')
    plt.close()



def run_exercise_1(args):
    
    
    # Load debug data
    print(f"Loading data from {args.debug_file}")
    try:
        X, Fs, expected_outputs = load_exercise1_data(args.debug_file)
    except FileNotFoundError:
        print(f"Error: File {args.debug_file} not found.")
        return
    
    # Prepare data for convolution
    X_ims, MX, Fs_flat, dimensions = prepare_data(X, Fs)
    
    # Print dimensions
    print("\nDimensions:")
    print(f"  Filter size (f): {dimensions['f']}")
    print(f"  Number of filters (nf): {dimensions['nf']}")
    print(f"  Number of images (n): {dimensions['n']}")
    print(f"  Patches per image (n_p): {dimensions['n_p']}")
    print(f"  X shape: {X.shape}")
    print(f"  Fs shape: {Fs.shape}")
    print(f"  X_ims shape: {X_ims.shape}")
    
    # Determine which methods to run
    if args.method == 'all':
        methods = ["dot", "matmul", "einsum"]
    else:
        methods = [args.method]
    
    # Run and time each method
    results = {}
    
    for method in methods:
        print(f"\nRunning {method} method...")
        
        # Time the method
        mean_time, std_time, outputs = time_method(
            method, X_ims, MX, Fs, Fs_flat, dimensions, runs=args.runs
        )
        
        # Verify outputs
        max_diff = verify_outputs(outputs, expected_outputs, method, dimensions)
        
        print(f"Max difference with expected outputs: {max_diff:.8f}")
        print(f"Average execution time: {mean_time:.4e}s ± {std_time:.2e}s")
        
        # Store results
        results[method] = {
            'mean_time': mean_time,
            'std_time': std_time,
            'max_diff': max_diff
        }
    
    # Compare methods if more than one was run
    if len(methods) > 1:
        print("\nMethod Comparison:")
        
        # Find fastest method
        fastest_method = min(results.keys(), key=lambda x: results[x]['mean_time'])
        
        print(f"Fastest method: {fastest_method} ({results[fastest_method]['mean_time']:.4e}s)")
        
        # Print speedup factors
        for method in methods:
            if method != fastest_method:
                speedup = results[method]['mean_time'] / results[fastest_method]['mean_time']
                print(f"{method} is {speedup:.2f}x slower than {fastest_method}")
    
    print("\nConclusion:")
    print("While the differences may not be significant with only a few images,")
    print("einsum tends to be much faster when processing large batches of images.")

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

def run_exercise_3(args):
    """
    Run Exercise 3: Train small networks with cyclic learning rates.
    
    Args:
        data_path: Path to CIFAR-10 data directory
        save_plots: Whether to save plots
        args: Command line arguments from argparse
    """
    
    # Set up logging to both file and console
    log_dir = "results/exercise3"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'architecture_{args.architecture}_train_log.txt')
    
    # Create a file handler for logging
    original_stdout = sys.stdout
    log_handle = open(log_file, 'w')
    sys.stdout = log_handle
    
    try:
        # Set number of threads for better performance
        os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
        os.environ["MKL_NUM_THREADS"] = str(args.num_threads)
        np.set_printoptions(precision=3)
        
        print(f"Running Exercise 3 with architecture {args.architecture}, {args.n_train} training samples, {args.num_threads} threads")
        
        # Load data
        print("Loading and preprocessing data...")
        try:
            # Use the improved data loading function
            X_train, Y_train, X_val, Y_val, X_test, Y_test = load_cifar_data(args.data_path, args.n_train)
            
            # Common parameters
            clr_params = {
                "n_batch": 100,
                "eta_min": 1e-5,
                "eta_max": 1e-1,
                "n_s": 800,  # Initial step size
                "n_cycles": 3
            }
            
            lambda_reg = 0.003  # L2 regularization parameter
            
            # Train specific architecture
            arch_params = {
                1: {"f": 2, "nf": 3, "nh": 50},
                2: {"f": 4, "nf": 10, "nh": 50},
                3: {"f": 8, "nf": 40, "nh": 50},
                4: {"f": 16, "nf": 160, "nh": 50}
            }
            
            selected_arch = arch_params[args.architecture]
            print(f"\nTraining Architecture {args.architecture}: f={selected_arch['f']}, nf={selected_arch['nf']}, nh={selected_arch['nh']}")
            
            model = ConvolutionalNetwork(
                f=selected_arch['f'],
                nf=selected_arch['nf'],
                nh=selected_arch['nh']
            )
            
            history = train_with_cyclical_lr(
                model, X_train, Y_train, X_val, Y_val,
                clr_params, lambda_reg,
                use_label_smoothing=False,
                logging_freq=200, verbose=True
            )
            
            test_loss, test_acc = model.compute_loss_and_accuracy(X_test, Y_test, lambda_reg)
            print(f"\nFinal test accuracy: {test_acc:.4f}")
            
            # Save training results
            save_ex3_results(args, history, test_acc, clr_params, lambda_reg)
            
            # Plot learning curves
            fig = plt.figure(figsize=(16, 12))

            # Define font sizes
            SMALL_SIZE = 16
            MEDIUM_SIZE = 20 
            BIGGER_SIZE = 24
            TITLE_SIZE = 28

            # First subplot
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(history["update_steps"], history["loss_train"], label="Training Loss", linewidth=3)
            ax1.plot(history["update_steps"], history["loss_val"], label="Validation Loss", linewidth=3)

            # Set font sizes explicitly for each element
            ax1.set_xlabel("Update Step", fontsize=MEDIUM_SIZE)
            ax1.set_ylabel("Loss", fontsize=MEDIUM_SIZE)
            ax1.set_title(f"Architecture {args.architecture} - Loss Curves", fontsize=TITLE_SIZE)
            ax1.tick_params(axis='both', which='major', labelsize=SMALL_SIZE)
            ax1.legend(fontsize=MEDIUM_SIZE, loc='upper right')
            ax1.set_ylim(1.0, 2.5)
            ax1.grid(True)

            # Second subplot
            ax2 = plt.subplot(2, 1, 2)
            ax2.plot(history["update_steps"], history["acc_train"], label="Training Accuracy", linewidth=3)
            ax2.plot(history["update_steps"], history["acc_val"], label="Validation Accuracy", linewidth=3)

            # Set font sizes explicitly for each element
            ax2.set_xlabel("Update Step", fontsize=MEDIUM_SIZE)
            ax2.set_ylabel("Accuracy", fontsize=MEDIUM_SIZE)
            ax2.set_title(f"Architecture {args.architecture} - Accuracy Curves", fontsize=TITLE_SIZE)
            ax2.tick_params(axis='both', which='major', labelsize=SMALL_SIZE)
            ax2.legend(fontsize=MEDIUM_SIZE, loc='lower right')
            ax2.set_ylim(0.1, 0.7)
            ax2.grid(True)

            # Adjust layout with more space
            plt.tight_layout(pad=3.0)

            # Save with explicit DPI setting
            save_path = f'results/exercise3/architecture_{args.architecture}_curves.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            print(f"\nPlot saved to: {save_path}")

            # Close the figure to free memory
            plt.close()
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print("Make sure the CIFAR-10 data is available in the specified path.")
            print("You can download it from https://www.cs.toronto.edu/~kriz/cifar.html")
    
    finally:
        # Restore stdout and close log file
        sys.stdout = original_stdout
        log_handle.close()
        print(f"\nTraining log saved to: {log_file}")

def run_exercise_4(args):
    """
    Run Exercise 4: Larger networks and regularization with label smoothing.
    
    Args:
        args: Command line arguments from argparse
    """
    print("Running Exercise 4: Larger Networks with Label Smoothing")
    print("-" * 50)
    
    try:
        # Set up log directory
        log_dir = "results/exercise4"
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up logging
        log_file = os.path.join(log_dir, 'exercise4_train_log.txt')
        original_stdout = sys.stdout
        log_handle = open(log_file, 'w')
        sys.stdout = log_handle
        
        # Extract parameters
        n_train = args.n_train
        num_threads = args.num_threads
        longer_training = args.longer
        epsilon = args.epsilon
        
        # Set threads for better performance
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)
        np.set_printoptions(precision=3)
        
        print(f"Running Exercise 4 with {n_train} training samples, {num_threads} threads")
        print(f"Longer training: {longer_training}, Label smoothing epsilon: {epsilon}")
        
        # Load data
        print("Loading and preprocessing data...")
        X_train, Y_train, X_val, Y_val, X_test, Y_test = load_cifar_data(args.data_path, n_train)
        
        # Common parameters
        clr_params = {
            "n_batch": 100,
            "eta_min": 1e-5,
            "eta_max": 1e-1,
            "n_s": 800,  # Initial step size
            "n_cycles": 4
        }
        
        lambda_reg = 0.0025  # L2 regularization parameter
        
        # Architecture 5 parameters
        f = 4
        nf = 40
        nh = 300
        
        print(f"\nTraining Architecture 5: f={f}, nf={nf}, nh={nh}")
        
        # Train without label smoothing
        print("Training without label smoothing...")
        model_no_smooth = ConvolutionalNetwork(f=f, nf=nf, nh=nh)
        history_no_smooth = train_with_cyclical_lr(
            model_no_smooth, X_train, Y_train, X_val, Y_val,
            clr_params, lambda_reg,
            use_label_smoothing=False,
            logging_freq=200, verbose=True
        )
        
        # Compute test accuracy without smoothing
        test_loss_no_smooth, test_acc_no_smooth = model_no_smooth.compute_loss_and_accuracy(X_test, Y_test, lambda_reg)
        print(f"\nFinal test accuracy without label smoothing: {test_acc_no_smooth:.4f}")
        
        # Save results without smoothing
        save_ex4_results(history_no_smooth, test_acc_no_smooth, 5, args, clr_params, lambda_reg, log_dir)
        
        # Train with label smoothing
        print("\nTraining with label smoothing...")
        model_smooth = ConvolutionalNetwork(f=f, nf=nf, nh=nh)
        history_smooth = train_with_cyclical_lr(
            model_smooth, X_train, Y_train, X_val, Y_val,
            clr_params, lambda_reg,
            use_label_smoothing=True,
            epsilon=epsilon,
            logging_freq=200, verbose=True
        )
        
        # Compute test accuracy with smoothing
        test_loss_smooth, test_acc_smooth = model_smooth.compute_loss_and_accuracy(X_test, Y_test, lambda_reg)
        print(f"\nFinal test accuracy with label smoothing: {test_acc_smooth:.4f}")
        
        # Save results with smoothing
        save_ex4_results(history_smooth, test_acc_smooth, 5, args, clr_params, lambda_reg, log_dir)
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        
        # Set font sizes
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12
        })
        
        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(history_no_smooth["update_steps"], history_no_smooth["loss_train"], label="Training Loss")
        plt.plot(history_no_smooth["update_steps"], history_no_smooth["loss_val"], label="Validation Loss")
        plt.xlabel("Update Step", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Without Label Smoothing - Loss Curves", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim(0.0, 2.5)

        
        plt.subplot(1, 2, 2)
        plt.plot(history_smooth["update_steps"], history_smooth["loss_train"], label="Training Loss")
        plt.plot(history_smooth["update_steps"], history_smooth["loss_val"], label="Validation Loss")
        plt.xlabel("Update Step", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("With Label Smoothing - Loss Curves", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim(0.0, 2.5)
        
        plt.tight_layout()
        
        if args.save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(log_dir, f'label_smoothing_comparison_{timestamp}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plots saved to {plot_path}")
        
        plt.close()
        
        # Print comparison summary
        print("\nComparison Summary:")
        print(f"Without Label Smoothing - Test Accuracy: {test_acc_no_smooth:.4f}")
        print(f"With Label Smoothing - Test Accuracy: {test_acc_smooth:.4f}")
        print(f"Improvement: {(test_acc_smooth - test_acc_no_smooth) * 100:.2f}%")
        
        print("\nExercise 4 completed successfully!")
        
        # Close log file
        sys.stdout = original_stdout
        log_handle.close()
        print(f"Training log saved to: {log_file}")
        
        return history_no_smooth, history_smooth, test_acc_no_smooth, test_acc_smooth
        
    except Exception as e:
        print(f"Error in Exercise 4: {e}")
        import traceback
        traceback.print_exc()
        
        # Restore stdout
        if 'original_stdout' in locals() and 'log_handle' in locals():
            sys.stdout = original_stdout
            log_handle.close()

def generate_architecture_comparison(data_path='./data/cifar-10-batches-py', save_plots=True, args=None):
    """
    Generate comparison plots for all architectures in Exercise 3.
    
    Args:
        data_path: Path to CIFAR-10 data directory
        save_plots: Whether to save plots
        args: Command line arguments from argparse
    """
    print("Generating Architecture Comparison Charts")
    print("-" * 50)
    
    try:
        # Define the log directory
        log_dir = "results/exercise3"
        os.makedirs(log_dir, exist_ok=True)
        
        # Architecture parameters
        arch_params = {
            1: {"f": 2, "nf": 3, "nh": 50, "title": "Arch 1: f=2, nf=3, nh=50"},
            2: {"f": 4, "nf": 10, "nh": 50, "title": "Arch 2: f=4, nf=10, nh=50"},
            3: {"f": 8, "nf": 40, "nh": 50, "title": "Arch 3: f=8, nf=40, nh=50"},
            4: {"f": 16, "nf": 160, "nh": 50, "title": "Arch 4: f=16, nf=160, nh=50"}
        }
        
        # Collect test accuracies and training times
        test_accuracies = []
        training_times = []
        labels = []
        
        for arch_id, arch_info in arch_params.items():
            # Try to load results from log file
            log_file = os.path.join(log_dir, f'architecture_{arch_id}_train_log.txt')
            
            if os.path.exists(log_file):
                print(f"Loading results for Architecture {arch_id}")
                test_acc, train_time = extract_results_from_log(log_file)
                
                if test_acc is not None and train_time is not None:
                    test_accuracies.append(test_acc)
                    training_times.append(train_time)
                    labels.append(arch_info["title"])
                    print(f"Architecture {arch_id}: Accuracy = {test_acc:.4f}, Time = {train_time:.2f}s")
            else:
                print(f"No results found for Architecture {arch_id}")
        
        if not test_accuracies:
            print("No results found for any architecture. Run Exercise 3 first.")
            return
            
        # Create bar charts
        plt.figure(figsize=(16, 6))
        
        # Test accuracy chart
        plt.subplot(1, 2, 1)
        bars = plt.bar(range(len(test_accuracies)), test_accuracies, tick_label=labels)
        plt.xlabel("Architecture")
        plt.ylabel("Test Accuracy")
        plt.title("Test Accuracy Comparison Across Architectures")
        plt.ylim(0, 1.0)
        plt.grid(True, axis='y')
        plt.xticks(rotation=45)
        
        # Add accuracy values on top of bars
        for bar, acc in zip(bars, test_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                     f"{acc:.4f}", ha='center', va='bottom')
        
        # Training time chart
        plt.subplot(1, 2, 2)
        bars = plt.bar(range(len(training_times)), training_times, tick_label=labels)
        plt.xlabel("Architecture")
        plt.ylabel("Training Time (seconds)")
        plt.title("Training Time Comparison Across Architectures")
        plt.grid(True, axis='y')
        plt.xticks(rotation=45)
        
        # Add time values on top of bars
        for bar, time in zip(bars, training_times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                     f"{time:.1f}s", ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(log_dir, 'architecture_comparison.png')
            plt.savefig(plot_path)
            print(f"Comparison charts saved to {plot_path}")
        
        plt.close()
        
        print("Architecture comparison completed successfully!")
        
    except Exception as e:
        print(f"Error in architecture comparison: {e}")
        import traceback
        traceback.print_exc()

def extract_results_from_log(log_file):
    """Extract test accuracy and training time from a log file."""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
            # Extract test accuracy
            acc_match = re.search(r"Final test accuracy: ([0-9.]+)", content)
            test_acc = float(acc_match.group(1)) if acc_match else None
            
            # Extract training time
            time_match = re.search(r"Training time: ([0-9.]+)", content)
            train_time = float(time_match.group(1)) if time_match else None
            
            return test_acc, train_time
    except Exception as e:
        print(f"Error extracting results from {log_file}: {e}")
        return None, None


def main():
    """
    Main function to run the exercises.
    """
    parser = argparse.ArgumentParser(description='Assignment 3: Convolutional Neural Networks')
    
    # Common arguments
    parser.add_argument('--exercise', type=int, choices=[1, 2, 3, 4, 5], default=3,
                        help='Exercise number to run (5 for architecture comparison)')
    parser.add_argument('--data_path', type=str, default='./data/cifar-10-batches-py',
                        help='Path to data directory')
    parser.add_argument('--debug_file', type=str, default='./data/debug_info.npz',
                        help='Path to debug data file')
    parser.add_argument('--save_plots', action='store_true', help='Save plots to disk', default=True)
    parser.add_argument('--num_threads', type=int, default=4,
                        help='Number of threads for numpy to use')
    parser.add_argument('--n_train', type=int, default=49000,
                        help='Number of training samples to use')
    
    # Exercise 1 specific arguments
    parser.add_argument('--method', type=str, choices=['all', 'dot', 'matmul', 'einsum'], 
                        default='all', help='Convolution method to use for Exercise 1')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of runs for timing in Exercise 1')
    
    # Exercise 3 specific arguments
    parser.add_argument('--architecture', type=int, choices=[1, 2, 3, 4, 6], default=2,
                        help='Architecture to train in Exercise 3')
    parser.add_argument('--increasing_cycle', action='store_true',
                        help='Use increasing cycle length for training')
    parser.add_argument('--wider_network', action='store_true',
                        help='Use wider network version for architecture 2')
    
    # Exercise 4 specific arguments
    parser.add_argument('--longer', action='store_true',
                        help='Run longer training with increasing step sizes')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Label smoothing factor (for Exercise 4)')
    
    args = parser.parse_args()
    
    # Import modules needed for regex
    import re
    
    print(f"Running Assignment 3 - Exercise {args.exercise}")
    print("-" * 70)
    print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of threads: {args.num_threads}")
    print("-" * 70)
    
    # Set number of threads for better performance
    os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.num_threads)
    np.set_printoptions(precision=3)
    
    # Run the selected exercise
    if args.exercise == 1:
        run_exercise_1(args)
    elif args.exercise == 2:
        run_exercise_2(args)
    elif args.exercise == 3:
        run_exercise_3(args)
    elif args.exercise == 4:
        run_exercise_4(args)
    elif args.exercise == 5:
        generate_architecture_comparison(args.data_path, args.save_plots, args)
    else:
        print("Please select an exercise to run (1-5)")
    
    print(f"\nAssignment 3 Exercise {args.exercise} completed!")

if __name__ == "__main__":
    main()