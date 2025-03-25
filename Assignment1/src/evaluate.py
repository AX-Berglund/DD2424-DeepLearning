import numpy as np

def ComputeAccuracy(P, y):
    """
    Computes the classification accuracy of the network.
    
    Parameters:
      P : K x N matrix of softmax probabilities (each column sums to 1)
      y : 1D array of length N containing ground truth class indices (0 to K-1)
    
    Returns:
      accuracy : Scalar value representing the accuracy percentage
    """
    # Get the predicted class index for each image (argmax over K classes)
    y_pred = np.argmax(P, axis=0)  # Shape: (N,)

    # Compute the percentage of correct predictions
    accuracy = np.mean(y_pred == y) * 100  # Convert to percentage
    return accuracy

def CompareGradients(dict1, dict2, tolerance=1e-6):
    """
    Compares two dictionaries containing gradients and returns the values
    where the absolute difference is greater than the specified tolerance.

    Parameters:
      dict1 : dict
          First dictionary containing gradients.
      dict2 : dict
          Second dictionary containing gradients.
      tolerance : float
          The threshold for considering two values as different.

    Returns:
      differences : dict
          A dictionary containing the keys and the corresponding values
          where the absolute difference is greater than the tolerance.
    """
    differences = {}
    for key in dict1:
        diff = np.abs(dict1[key] - dict2[key])
        if np.any(diff > tolerance):
            differences[key] = diff[diff > tolerance]

    if len(differences) == 0:
        print("Gradients match within the specified tolerance.")
        return None
    else:
        print(f"Number of gradients that differ: {len(differences)}")
        print(f"Max difference: {np.max([np.max(diff) for diff in differences.values()])}")

        return differences