#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path='./data/cifar-10-batches-py/data_batch_1', normalize=True):
    """
    Load CIFAR-10 data from the specified file.
    
    Args:
        file_path: Path to the data file
        normalize: Whether to normalize the data
        
    Returns:
        X: Data, shape (3072, n)
        Y: One-hot encoded labels, shape (10, n)
        y: Class labels, shape (n,)
    """
    # Load data
    import pickle
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    
    # Extract data and labels
    X = data_dict[b'data'].T  # Transpose to shape (3072, n)
    y = np.array(data_dict[b'labels'])
    
    # Normalize data if required
    if normalize:
        X = X / 255.0
    
    # Convert labels to one-hot encoding
    Y = np.zeros((10, y.size))
    Y[y, np.arange(y.size)] = 1
    
    return X, Y, y

def load_debug_data(file_path='./data/debug_info.npz'):
    """
    Load debugging data for exercises.
    
    Args:
        file_path: Path to the debug data file
        
    Returns:
        X: Input data
        Fs: Filters
        conv_outputs: Expected convolution outputs
    """
    # Load the data file
    load_data = np.load(file_path, allow_pickle=True)
    
    # Extract the required data
    X = load_data['X']
    Fs = load_data['Fs']
    conv_outputs = load_data['conv_outputs']
    
    return X, Fs, conv_outputs

def verify_gradients(computed_grads, expected_grads):
    """
    Verify computed gradients against expected values.
    
    Args:
        computed_grads: Dictionary of computed gradients
        expected_grads: Dictionary of expected gradients
        
    Returns:
        dict: Dictionary containing verification results
    """
    results = {}
    
    for key in expected_grads:
        if key in computed_grads:
            # Compute differences
            diff = np.abs(computed_grads[key] - expected_grads[key])
            results[key] = {
                'max_diff': np.max(diff),
                'mean_diff': np.mean(diff),
                'median_diff': np.median(diff),
                'std_diff': np.std(diff)
            }
    
    return results

def detailed_gradient_verification(computed_grads, expected_grads):
    """
    Perform detailed verification of computed gradients.
    
    Args:
        computed_grads: Dictionary of computed gradients
        expected_grads: Dictionary of expected gradients
        
    Returns:
        str: Detailed verification report
    """
    report = []
    
    for key in expected_grads:
        if key in computed_grads:
            # Check shapes
            report.append(f"Checking {key} shapes...")
            comp_shape = computed_grads[key].shape
            exp_shape = expected_grads[key].shape
            report.append(f"Computed shape: {comp_shape}")
            report.append(f"Expected shape: {exp_shape}")
            
            if comp_shape != exp_shape:
                report.append(f"ERROR: Shape mismatch for {key}!")
                continue
            
            # Compute element-wise differences
            diff = computed_grads[key] - expected_grads[key]
            abs_diff = np.abs(diff)
            
            report.append("\nGradient difference statistics:")
            report.append(f"Maximum absolute difference: {np.max(abs_diff):.10f}")
            report.append(f"Average absolute difference: {np.mean(abs_diff):.10f}")
            report.append(f"Median absolute difference: {np.median(abs_diff):.10f}")
            report.append(f"Standard deviation of differences: {np.std(abs_diff):.10f}")
            
            # Check ranges
            report.append("\nChecking for potential numerical issues...")
            report.append(f"Range of computed gradients: [{np.min(computed_grads[key]):.4f}, {np.max(computed_grads[key]):.4f}]")
            report.append(f"Range of expected gradients: [{np.min(expected_grads[key]):.4f}, {np.max(expected_grads[key]):.4f}]")
            
            # Find locations of largest differences
            worst_indices = np.argsort(abs_diff.flatten())[-5:][::-1]
            report.append("\nTop 5 worst differences:")
            for idx in worst_indices:
                i, j = np.unravel_index(idx, abs_diff.shape)
                report.append(f"Position ({i},{j}):")
                report.append(f"  Computed value: {computed_grads[key][i,j]:.10f}")
                report.append(f"  Expected value: {expected_grads[key][i,j]:.10f}")
                report.append(f"  Absolute difference: {abs_diff[i,j]:.10f}")
            
            report.append('-' * 50)
    
    return '\n'.join(report)

def plot_learning_curves(history, title=None, figsize=(12, 5)):
    """
    Plot learning curves from training history.
    
    Args:
        history: Dictionary containing training history
        title: Plot title
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss curves
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Evaluation Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy curves
    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Evaluation Step')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    ax2.grid(True)
    
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    
    return fig

def plot_learning_rate(history, figsize=(10, 5)):
    """
    Plot learning rate schedule from training history.
    
    Args:
        history: Dictionary containing training history
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(history['learning_rates'])
    plt.xlabel('Update Step')
    plt.ylabel('Learning Rate')
    plt.title('Cyclical Learning Rate Schedule')
    plt.grid(True)
    
    return plt.gcf()

def plot_architecture_comparison(architectures, figsize=(12, 5)):
    """
    Plot comparison of different architectures.
    
    Args:
        architectures: List of dictionaries with architecture details
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Extract data
    names = [arch['name'] for arch in architectures]
    accuracies = [arch['val_acc'] for arch in architectures]
    times = [arch['training_time'] for arch in architectures]
    
    # Plot accuracy comparison
    ax1.bar(names, accuracies, color='steelblue')
    ax1.set_xlabel('Architecture')
    ax1.set_ylabel('Validation Accuracy')
    ax1.set_title('Accuracy Comparison')
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 0.01, f"{v:.3f}", ha='center')
    
    # Plot training time comparison
    ax2.bar(names, times, color='indianred')
    ax2.set_xlabel('Architecture')
    ax2.set_ylabel('Training Time (s)')
    ax2.set_title('Training Time Comparison')
    for i, v in enumerate(times):
        ax2.text(i, v + 5, f"{v:.1f}s", ha='center')
    
    plt.tight_layout()
    
    return fig