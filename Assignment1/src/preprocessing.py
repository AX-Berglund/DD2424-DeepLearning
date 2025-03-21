import numpy as np

def reshape_images(images):
    """
    Reshape CIFAR-10 images from (10000, 3072) to (10000, 32, 32, 3).

    Parameters:
        images (numpy.ndarray): Flattened CIFAR-10 image data (10000, 3072).

    Returns:
        reshaped_images (numpy.ndarray): Images reshaped to (10000, 32, 32, 3).
    """
    # Reshape to (num_samples, 3, 32, 32)
    images = images.reshape((len(images), 3, 32, 32))

    # Transpose to (num_samples, 32, 32, 3) for correct image format
    reshaped_images = images.transpose(0, 2, 3, 1)

    return reshaped_images

def normalize_images(images):
    """
    Normalize image pixel values from range [0, 255] to [0, 1].

    Parameters:
        images (numpy.ndarray): Image data.

    Returns:
        normalized_images (numpy.ndarray): Normalized image data.
    """
    return images.astype(np.float32) / 255.0
