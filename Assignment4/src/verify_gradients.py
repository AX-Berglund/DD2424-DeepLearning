import numpy as np
import matplotlib.pyplot as plt
from src.utils import init_rng
from src.data import read_data, create_mappings
from src.gradient_check import check_gradients, torch_gradient_check

def main():
    """
    Run gradient checking to verify the RNN implementation.
    """
    # Set paths
    data_dir = 'data'
    book_fname = 'data/goblet_book.txt'
    
    # Read data
    print("Reading book data...")
    book_data = read_data(book_fname)
    print(f"Book length: {len(book_data)} characters")
    
    # Create character mappings
    char_to_ind, ind_to_char, unique_chars = create_mappings(book_data)
    K = len(unique_chars)
    print(f"Number of unique characters: {K}")
    
    # Run gradient checking
    print("\nRunning gradient checking...")
    
    # 1. Check analytic gradients vs numerical gradients
    print("1. Checking analytic vs numerical gradients...")
    is_correct_num = check_gradients(book_data, char_to_ind)
    print(f"Gradient check with numerical gradients: {'PASSED' if is_correct_num else 'FAILED'}")
    
    # 2. Check analytic gradients vs PyTorch gradients
    print("\n2. Checking analytic vs PyTorch gradients...")
    is_correct_torch = torch_gradient_check(book_data, char_to_ind)
    print(f"Gradient check with PyTorch: {'PASSED' if is_correct_torch else 'FAILED'}")
    
    if is_correct_num and is_correct_torch:
        print("\nGradient checking PASSED! The gradient computations are correct.")
    else:
        print("\nGradient checking FAILED! Please check your implementation.")

if __name__ == "__main__":
    main()