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

def visualize_W_matrix(trained_net, save_path="../results/images/W1_matrix.png"):
    """
    Visualizes the learnt W matrix as class template images.

    Parameters:
        trained_net : dict
            Dictionary containing the trained network parameters.
        class_names : list
            List of class names corresponding to the CIFAR-10 dataset.
        save_path : str
            Path to save the visualized W matrix image.
    """
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    plt.figure(figsize=(20, 4))  # Increase the figure size
    plt.suptitle("The learnt W matrix visualized as class template images.", fontsize=16, fontweight='bold')

    Ws = trained_net['W'].T.reshape((32, 32, 3, 10), order='F')
    W_im = np.transpose(Ws, (1, 0, 2, 3))

    for i in range(10):
        w_im = W_im[:, :, :, i]
        w_im_norm = (w_im - np.min(w_im)) / (np.max(w_im) - np.min(w_im))
        plt.subplot(1, 10, i + 1)
        plt.imshow(w_im_norm, interpolation='nearest')
        plt.axis('off')
        plt.title(class_names[i], fontsize=15, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust layout to fit the title
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")


