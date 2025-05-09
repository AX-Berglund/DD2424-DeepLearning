#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
import argparse

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

def load_debug_data(file_path):
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


def main():

    parser = argparse.ArgumentParser(description='Exercise 1: Convolutional Neural Networks')
    parser.add_argument('--method', type=str, choices=['all', 'dot', 'matmul', 'einsum'], 
                        default='all', help='Convolution method to use for Exercise 1')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of runs for timing in Exercise 1')
    args = parser.parse_args()

    # Load debug data
    print(f"Loading data from {args.debug_file}")
    try:
        X, Fs, expected_outputs = load_debug_data(args.debug_file)
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


if __name__ == "__main__":
    main()