import numpy as np

def read_data(file_path):
    """
    Read the training text data from file.
    
    Args:
        file_path: Path to the text file
    
    Returns:
        book_data: String containing the book text
    """
    with open(file_path, "r", encoding="utf-8") as fid:
        book_data = fid.read()
    return book_data

def create_mappings(book_data):
    """
    Create character to index and index to character mappings.
    
    Args:
        book_data: String containing the book text
    
    Returns:
        char_to_ind: Dictionary mapping characters to indices
        ind_to_char: Dictionary mapping indices to characters
        unique_chars: List of unique characters
    """
    # Find all unique characters in the book
    unique_chars = list(set(book_data))
    K = len(unique_chars)
    
    # Create dictionaries for char to index and index to char mappings
    char_to_ind = {char: i for i, char in enumerate(unique_chars)}
    ind_to_char = {i: char for i, char in enumerate(unique_chars)}
    
    return char_to_ind, ind_to_char, unique_chars

def get_one_hot_encoding(sequence, char_to_ind, K):
    """
    Convert a sequence of characters to one-hot encoded vectors.
    
    Args:
        sequence: String of characters
        char_to_ind: Dictionary mapping characters to indices
        K: Number of unique characters
    
    Returns:
        one_hot: One-hot encoded matrix, shape (K, len(sequence))
    """
    indices = [char_to_ind[char] for char in sequence]
    one_hot = np.zeros((K, len(sequence)))
    one_hot[indices, np.arange(len(sequence))] = 1
    return one_hot

def get_sequence_data(book_data, e, seq_length, char_to_ind, K):
    """
    Get a training sequence from the book data.
    
    Args:
        book_data: String containing the book text
        e: Starting index for the sequence
        seq_length: Length of the sequence
        char_to_ind: Dictionary mapping characters to indices
        K: Number of unique characters
    
    Returns:
        X: Input sequence, shape (K, seq_length)
        Y: Target sequence, shape (K, seq_length)
        X_chars: Input characters
        Y_chars: Target characters
    """
    X_chars = book_data[e:e+seq_length]
    Y_chars = book_data[e+1:e+seq_length+1]
    
    X = get_one_hot_encoding(X_chars, char_to_ind, K)
    Y = get_one_hot_encoding(Y_chars, char_to_ind, K)
    
    return X, Y, X_chars, Y_chars