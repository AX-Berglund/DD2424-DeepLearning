import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

def load_cifar10_batch(batch_id):
    """
    Load a single batch of CIFAR-10 data.

    Parameters:
        batch_id (int): The batch number (1 to 5) to load.

    Returns:
        images (numpy.ndarray): Flattened image data of shape (10000, 3072).
        labels (numpy.ndarray): Corresponding labels of shape (10000,).
    """
    cifar10_path = "/Users/axhome/AX/MASTER/Courses/DD2424/DD2424-DeepLearning/Assignment1/data/cifar-10/cifar-10-batches-py"
    # Construct full file path
    batch_file = os.path.join(cifar10_path, f"data_batch_{batch_id}")

    # Load the batch file using pickle
    with open(batch_file, mode='rb') as file:
        batch = pickle.load(file, encoding='bytes')  # Ensures compatibility with Python 3

    # Extract image data and labels
    images = np.array(batch[b'data'])  # Image data in shape (num_samples, 3072)
    labels = np.array(batch[b'labels'])  # Labels as a 1D array

    return images, labels


def LoadBatch(batch_id, dtype=np.float64):
    """
    Load a CIFAR-10 batch file and return image data, one-hot labels, and raw labels.

    Parameters:
        filename (str): Path to the CIFAR-10 batch file.
        dtype (type): Data type for image and one-hot encoded label arrays (float32 or float64).

    Returns:
        X (numpy.ndarray): Image data of shape (3072, 10000), type float32/float64, values in [0,1].
        Y (numpy.ndarray): One-hot encoded labels of shape (10, 10000), type float32/float64.
        y (numpy.ndarray): Label vector of shape (10000,), type int (values 0-9).
    """

    cifar10_path = "/Users/axhome/AX/MASTER/Courses/DD2424/DD2424-DeepLearning/Assignment1/data/cifar-10/cifar-10-batches-py"
    # Construct full file path
    ## if batch_id is string
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

    # Convert labels to one-hot encoding
    K = 10  # Number of classes in CIFAR-10
    Y = np.zeros((K, X.shape[1]), dtype=dtype)  # Shape (10, 10000)
    Y[labels, np.arange(X.shape[1])] = 1  # Assign 1s for correct labels

    # Return X (3072×10000), Y (10×10000), y (10000,)

    # make sure labels are (1x10000)
    labels = labels.reshape(1, len(labels))

    return X, Y, labels
