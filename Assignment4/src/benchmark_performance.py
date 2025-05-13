import os
import numpy as np
import time
import matplotlib.pyplot as plt
from src.utils import init_rng
from src.data import read_data, create_mappings
from src.gradient_check import forward_pass, backward_pass
from src.optimization import forward_pass_optimized, backward_pass_optimized, benchmark_performance

def main():
    """
    Benchmark the performance of standard vs optimized implementations.
    """
    # Set paths
    data_dir = 'data'
    book_fname = os.path.join(data_dir, 'goblet_book.txt')
    
    # Initialize random number generator
    rng = init_rng(seed=400)
    
    # Read data
    print("Reading book data...")
    book_data = read_data(book_fname)
    
    # Create character mappings
    char_to_ind, ind_to_char, unique_chars = create_mappings(book_data)
    K = len(unique_chars)
    
    # Set hyperparameters
    m = 100  # Hidden state dimension
    seq_length = 25  # Sequence length
    
    # Create test data
    X = np.zeros((K, seq_length))
    for t in range(seq_length):
        idx = np.random.randint(0, K)
        X[idx, t] = 1
    
    # Create RNN parameters
    RNN = {
        'b': np.zeros((m, 1)),
        'c': np.zeros((K, 1)),
        'U': (1/np.sqrt(2*K)) * rng.standard_normal(size=(m, K)),
        'W': (1/np.sqrt(2*m)) * rng.standard_normal(size=(m, m)),
        'V': (1/np.sqrt(m)) * rng.standard_normal(size=(K, m))
    }
    
    # Initial hidden state
    h0 = np.zeros((m, 1))
    
    # Run benchmarks
    print("\nRunning performance benchmarks...")
    
    # Number of runs for each test
    num_runs = 100
    
    # Benchmark results
    results = benchmark_performance(X, h0, RNN, num_runs=num_runs,
                                  standard_forward=forward_pass, standard_backward=backward_pass)
    
    # Print results
    print("\nPerformance Results:")
    print("-" * 50)
    
    print("\nForward Pass:")
    print(f"Standard implementation: {results['forward']['standard']:.4f} seconds for {num_runs} runs")
    print(f"Optimized implementation: {results['forward']['optimized']:.4f} seconds for {num_runs} runs")
    print(f"Speedup: {results['forward']['standard'] / results['forward']['optimized']:.2f}x")
    print(f"Results match: {results.get('forward_match', 'N/A')}")
    
    print("\nBackward Pass:")
    print(f"Standard implementation: {results['backward']['standard']:.4f} seconds for {num_runs} runs")
    print(f"Optimized implementation: {results['backward']['optimized']:.4f} seconds for {num_runs} runs")
    print(f"Speedup: {results['backward']['standard'] / results['backward']['optimized']:.2f}x")
    print(f"Results match: {results.get('backward_match', 'N/A')}")
    
    # Total speedup
    std_total = results['forward']['standard'] + results['backward']['standard']
    opt_total = results['forward']['optimized'] + results['backward']['optimized']
    print(f"\nTotal combined speedup: {std_total / opt_total:.2f}x")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    # Forward pass
    plt.subplot(1, 2, 1)
    plt.bar(['Standard', 'Optimized'], 
            [results['forward']['standard'], results['forward']['optimized']])
    plt.title('Forward Pass Time')
    plt.ylabel('Time (seconds)')
    
    # Backward pass
    plt.subplot(1, 2, 2)
    plt.bar(['Standard', 'Optimized'], 
            [results['backward']['standard'], results['backward']['optimized']])
    plt.title('Backward Pass Time')
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    print("\nBenchmark results saved to 'benchmark_results.png'")
    
    print("\nOptimization strategies implemented:")
    print("1. Pre-computing UX for all time steps at once")
    print("2. Using specialized code for sparse one-hot encoded inputs")
    print("3. Using np.outer() instead of np.matmul() for outer products")
    print("4. Computing all outputs at once after the time loop")
    print("5. Applying batched matrix operations where possible")

if __name__ == "__main__":
    main()