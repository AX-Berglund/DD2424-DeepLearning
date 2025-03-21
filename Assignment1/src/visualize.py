import numpy as np
import matplotlib.pyplot as plt

def plot_random_images(images, labels, num_images=10):
    """
    Display a set of random images from the CIFAR-10 dataset.

    Parameters:
        images (numpy.ndarray): CIFAR-10 images of shape (num_samples, 32, 32, 3).
        labels (numpy.ndarray): Corresponding labels.
        num_images (int): Number of random images to display.
    """
    # Define CIFAR-10 class names for better visualization
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    # Select random indices to display
    indices = np.random.choice(images.shape[0], num_images, replace=False)

    # Create subplots for visualization
    fig, axs = plt.subplots(1, num_images, figsize=(15, 3))
    
    for i, idx in enumerate(indices):
        axs[i].imshow(images[idx])
        axs[i].axis('off')
        axs[i].set_title(class_names[labels[idx]])  # Display class name

    plt.show()
