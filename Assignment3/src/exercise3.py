import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pickle
import argparse
import sys
from datetime import datetime
from tqdm import tqdm

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Exercise 3 - CNN with Patchify Layer')
    parser.add_argument('--architecture', type=int, choices=[1, 2, 3, 4], default=2,
                        help='Architecture to train (1-4 for basic architectures)')
    parser.add_argument('--n_train', type=int, default=49000,
                        help='Number of training samples to use')
    parser.add_argument('--data_path', type=str, default='../data/cifar-10-batches-py',
                        help='Path to CIFAR-10 data')
    parser.add_argument('--num_threads', type=int, default=4,
                        help='Number of threads for numpy to use')
    
    args = parser.parse_args()
    
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
            save_training_results(history, test_acc, args.architecture)
            

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
            ax1.set_ylim(1.0, 3.25)
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