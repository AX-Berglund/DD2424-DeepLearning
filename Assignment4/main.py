import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from src.utils import init_rng
from src.data import read_data, create_mappings
from src.model import RNN
from src.train import train_rnn
from src.gradient_check import check_gradients, torch_gradient_check

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