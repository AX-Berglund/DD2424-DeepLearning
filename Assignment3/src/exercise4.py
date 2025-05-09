import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pickle
import argparse
import sys
from datetime import datetime
from tqdm import tqdm
from src.exercise3 import ConvolutionalNetwork, load_cifar_data, train_with_cyclical_lr


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Exercise 4 - Label Smoothing Comparison')
    parser.add_argument('--n_train', type=int, default=49000,
                        help='Number of training samples to use')
    parser.add_argument('--data_path', type=str, default='../data/cifar-10-batches-py',
                        help='Path to CIFAR-10 data')
    parser.add_argument('--longer', action='store_true',
                        help='Run longer training with increasing step sizes')
    parser.add_argument('--num_threads', type=int, default=4,
                        help='Number of threads for numpy to use')
    
    args = parser.parse_args()
    
    # Set up logging to both file and console
    log_dir = "results/exercise4"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'exercise4_train_log.txt')
    
    # Create a file handler for logging
    original_stdout = sys.stdout
    log_handle = open(log_file, 'w')
    sys.stdout = log_handle
    
    try:
        # Set number of threads for better performance
        os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
        os.environ["MKL_NUM_THREADS"] = str(args.num_threads)
        np.set_printoptions(precision=3)
        
        print(f"Running Exercise 4 with {args.n_train} training samples, {args.num_threads} threads")
        
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
                "n_cycles": 4
            }
            
            lambda_reg = 0.0025  # L2 regularization parameter
            
            if args.longer:
                print("\nRunning longer training with increasing step sizes...")
                train_longer_with_increasing_steps(
                    X_train, Y_train, X_val, Y_val, X_test, Y_test, clr_params, lambda_reg
                )
            else:
                print("\nRunning label smoothing comparison...")
                compare_label_smoothing(
                    X_train, Y_train, X_val, Y_val, X_test, Y_test, clr_params, lambda_reg
                )
                
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