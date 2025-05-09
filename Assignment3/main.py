#!/usr/bin/env python
# coding: utf-8

import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import os
import sys
from datetime import datetime
from tqdm import tqdm

from src.exercise1 import (
    prepare_data, 
    time_method,
    verify_outputs,
    load_debug_data as load_exercise1_data
)
from src.exercise2 import (
    forward_pass,
    backward_pass,
    compute_loss,
    load_debug_data as load_exercise2_data,
    verify_gradients
)
from src.exercise3 import ConvolutionalNetwork, load_cifar_data, train_with_cyclical_lr
from src.exercise3 import save_training_results as save_ex3_results
from src.exercise4 import save_training_results as save_ex4_results

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