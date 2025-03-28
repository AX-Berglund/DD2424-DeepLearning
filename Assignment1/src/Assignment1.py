import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle
import os


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

import numpy as np


def ComputeLoss(P, y, W, lam):
    """
    Computes the cost function J using softmax probabilities.
    
    Parameters:
      P   : K x N matrix of softmax probabilities
      y   : 1D array of length N with ground truth class indices (integers 0 to 9)
      W   : K x d weight matrix (needed for regularization term)
      lam : Regularization coefficient (float)
    
    Returns:
      J : Scalar value representing the total loss (cross-entropy + regularization)
    """
    # Number of training examples
    N = P.shape[1]

    # Pick correct class probabilities for each example using indexing
    # Example P[y=5,Sample1] is the probability of class 5 for the first example
    log_probs = -np.log(P[y, np.arange(N)] + 1e-15)  # Avoid log(0) 
    
    # Cross-entropy loss (mean over all examples)
    cross_entropy_loss = np.mean(log_probs)

    # Regularization term (L2 penalty on W)
    reg_term = lam * np.sum(W**2)

    # Total cost
    J = cross_entropy_loss + reg_term
    
    return J

# Utility Functions
def softmax(S):
    """Computes column-wise softmax of S, ensuring numerical stability."""
    S_shifted = S - np.max(S, axis=0, keepdims=True)
    exp_S = np.exp(S_shifted)
    return exp_S / np.sum(exp_S, axis=0, keepdims=True)

def ApplyNetwork(X, net):
    """
    Applies the network function: Computes class scores and applies softmax.
    
    Parameters:
      X   : d x N input images (each column is an image)
      net : Dictionary containing network parameters {"W": Weight matrix, "b": Bias vector}
    
    Returns:
      P : K x N matrix of class probabilities after softmax
    """
    return softmax(net["W"] @ X + net["b"])

def BackwardPass(X, Y, P, network, lam):
    """
    Computes gradients of the cost function J with respect to W and b.

    Parameters:
      X      : d x N input data matrix (each column is an image)
      Y      : K x N one-hot ground truth labels
      P      : K x N softmax probabilities from ApplyNetwork
      network: Dictionary containing model parameters:
               - "W": K x d weight matrix
               - "b": K x 1 bias vector
      lam    : Regularization coefficient

    Returns:
      grads: Dictionary with gradients
             - "W": K x d gradient matrix
             - "b": K x 1 gradient vector
    """
    N = X.shape[1]  # Number of samples in mini-batch

    # Compute gradient of the loss w.r.t. S (logits)
    G = P - Y  # (K x N)

    # Compute gradients
    grad_W = (1 / N) * (G @ X.T) + 2 * lam * network["W"]  # (K x d)
    grad_b = (1 / N) * np.sum(G, axis=1, keepdims=True)     # (K x 1)

    # Store gradients in a dictionary
    grads = {
        "W": grad_W,
        "b": grad_b
    }
    
    return grads


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

def preprocess_data(X_train_raw, X_val_raw, X_test_raw):
    """
    Normalizes the dataset based on training set mean and standard deviation.
    
    Parameters:
      X_train_raw, X_val_raw, X_test_raw: Raw data matrices
    
    Returns:
      X_train, X_val, X_test: Normalized datasets
    """
    X_train_mean = np.mean(X_train_raw, axis=1, keepdims=True)
    X_train_std = np.std(X_train_raw, axis=1, keepdims=True)

    X_train = (X_train_raw - X_train_mean) / X_train_std
    X_val = (X_val_raw - X_train_mean) / X_train_std
    X_test = (X_test_raw - X_train_mean) / X_train_std

    return X_train, X_val, X_test

def initialize_network(K=10, d=3072, seed=42):
    """
    Initializes the network parameters.
    
    Parameters:
      K   : Number of classes
      d   : Input dimension
      seed: Random seed for reproducibility
    
    Returns:
      init_net : Dictionary containing initialized parameters {"W": weight matrix, "b": bias vector}
    """
    rng = np.random.default_rng(seed)
    return {"W": 0.01 * rng.standard_normal(size=(K, d)), "b": np.zeros((K, 1))}


def MiniBatchGD(X, Y, GDparams, init_net, lam, X_val=None, Y_val=None, track_loss=False):
    """
    Performs Mini-Batch Gradient Descent to train the network.

    Parameters:
      X        : d x N input data matrix (each column is an image)
      Y        : K x N one-hot encoded ground truth labels
      GDparams : Dictionary containing {"n_batch": batch size, "eta": learning rate, "n_epochs": epochs}
      init_net : Dictionary containing initial parameters {"W": weight matrix, "b": bias vector}
      lam      : Regularization coefficient
      X_val, Y_val : Optional validation set
      track_loss : If True, track training and validation loss

    Returns:
      trained_net : Dictionary with the learned parameters
      loss_hist   : Dictionary with lists of training/validation losses per epoch (if track_loss=True)
      cost_hist   : Dictionary with lists of training/validation costs per epoch (if track_loss=True)
    """
    trained_net = copy.deepcopy(init_net)
    n_batch, eta, n_epochs = GDparams["n_batch"], GDparams["eta"], GDparams["n_epochs"]
    N = X.shape[1]
    rng = np.random.default_rng(seed=42)

    if track_loss:
        loss_hist = {"train": [], "val": []}
        cost_hist = {"train": [], "val": []}

    for epoch in range(n_epochs):
        shuffled_indices = rng.permutation(N)
        X_shuffled, Y_shuffled = X[:, shuffled_indices], Y[:, shuffled_indices]

        for j in range(N // n_batch):
            j_start, j_end = j * n_batch, (j + 1) * n_batch
            Xbatch, Ybatch = X_shuffled[:, j_start:j_end], Y_shuffled[:, j_start:j_end]

            P = ApplyNetwork(Xbatch, trained_net)
            grads = BackwardPass(Xbatch, Ybatch, P, trained_net, lam)

            trained_net["W"] -= eta * grads["W"]
            trained_net["b"] -= eta * grads["b"]

        if track_loss:
            # Full epoch evaluation
            P_train = ApplyNetwork(X, trained_net)
            train_loss = ComputeLoss(P_train, np.argmax(Y, axis=0), trained_net["W"], 0)
            train_cost = ComputeLoss(P_train, np.argmax(Y, axis=0), trained_net["W"], lam)
            loss_hist["train"].append(train_loss)
            cost_hist["train"].append(train_cost)

            if X_val is not None and Y_val is not None:
                P_val = ApplyNetwork(X_val, trained_net)
                val_loss = ComputeLoss(P_val, np.argmax(Y_val, axis=0), trained_net["W"], 0)
                val_cost = ComputeLoss(P_val, np.argmax(Y_val, axis=0), trained_net["W"], lam)
                loss_hist["val"].append(val_loss)
                cost_hist["val"].append(val_cost)

    if track_loss:
        return trained_net, loss_hist, cost_hist
    return trained_net


def main():
    # Load and preprocess data
    X_train_raw, Y_train, y_train = LoadBatch(1)
    X_val_raw, Y_val, y_val = LoadBatch(2)
    X_test_raw, Y_test, y_test = LoadBatch("test_batch")

    X_train, X_val, X_test = preprocess_data(X_train_raw, X_val_raw, X_test_raw)

    # Initialize network
    init_net = initialize_network()

    # Forward propagation using Softmax
    P = ApplyNetwork(X_train[:, :20], init_net)

    # Compute loss and accuracy
    loss = ComputeLoss(P, y_train[0][:20], init_net["W"], 0.1)
    accuracy = ComputeAccuracy(P, y_train[0][:20])

    # Compute gradients and compare with PyTorch
    grads_own = BackwardPass(X_train[:, :20], Y_train[:, :20], P, init_net, 0.1)
    grads_torch = BackwardPass(X_train[:, :20], Y_train[:, :20], P, init_net, 0.1)
    CompareGradients(grads_own, grads_torch)

    # Train the network
    GDparams = {"n_batch": 100, "eta": 0.001, "n_epochs": 10}
    trained_net = MiniBatchGD(X_train, Y_train, GDparams, init_net, 0.1)

    # Evaluate the trained network
    P_train, P_val, P_test = ApplyNetwork(X_train, trained_net), ApplyNetwork(X_val, trained_net), ApplyNetwork(X_test, trained_net)

    # Compute accuracy
    print("\nTraining Accuracy:", ComputeAccuracy(P_train, y_train[0]))
    print("Validation Accuracy:", ComputeAccuracy(P_val, y_val[0]))
    print("Testing Accuracy:", ComputeAccuracy(P_test, y_test[0]))


def evaluate(network, X, Y, y, lam=0.0):
    """
    Evaluate the model on given data.

    Parameters:
        network: dict containing 'W' and 'b'
        X: d x n input data
        Y: K x n one-hot encoded labels
        y: n-length vector of class indices (int)
        lam: regularization coefficient (default 0.0)

    Returns:
        loss (float), accuracy (float)
    """
    P = ApplyNetwork(X, network)
    
    W = network['W']

    # Cross-entropy loss
    cross_entropy_loss = ComputeLoss(P, y, W, lam)
    
    # Regularization term
    reg_term = lam * np.sum(W ** 2)
    cost = cross_entropy_loss + reg_term

    # Accuracy
    acc = ComputeAccuracy(P, y)

    print(f"Cost: {cost:.4f}, Accuracy: {acc:.2f}%")
    return cost, acc




def plot_curve(loss_hist, n_batch, eta, n_epochs, label, save_path):
    """
    Plots and saves the training and validation loss curves.

    Parameters:
        loss_hist (dict): Dictionary with keys "train" and "val" containing lists of losses.
        n_batch (int): Batch size used during training.
        eta (float): Learning rate.
        n_epochs (int): Number of training epochs.
        save_path (str): Path to save the plot (e.g. "../results/exercise_2/01/loss_curve_ALLBATCHES.png").
    """
    plt.figure()
    plt.plot(loss_hist["train"], label=f"Train {label}")
    plt.plot(loss_hist["val"], label=f"Val {label}")
    plt.title(f"Loss Curve (n_batch: {n_batch}, eta: {eta}, n_epochs: {n_epochs})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def visualize_weights(W, n_batch, eta, n_epochs, save_path):
    """
    Visualizes the learned weights as images and saves the figure.

    Parameters:
        W (ndarray): Weight matrix of shape (num_classes, input_dim).
        class_names (list of str): Names of the classes.
        n_batch (int): Batch size used during training.
        eta (float): Learning rate.
        n_epochs (int): Number of training epochs.
        save_path (str): Path to save the plot (e.g. "../results/exercise_2/01/weights_ALLBATCHES.png").
    """
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    Ws = W.T.reshape((32, 32, 3, 10), order='F')  # Assuming CIFAR-10 shape
    W_img = np.transpose(Ws, (1, 0, 2, 3))        # Shape: (32, 32, 3, 10)

    fig, axs = plt.subplots(1, 10, figsize=(15, 2))
    for i in range(10):
        w_img = W_img[:, :, :, i]
        w_img_norm = (w_img - np.min(w_img)) / (np.max(w_img) - np.min(w_img))
        axs[i].imshow(w_img_norm)
        axs[i].axis('off')
        axs[i].set_title(class_names[i], fontsize=8, fontweight='bold')

    plt.suptitle(f"Weight Visualization (n_batch: {n_batch}, eta: {eta}, n_epochs: {n_epochs})")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def exercise_1():
    # Load and preprocess data
    X_train_raw, Y_train, y_train = LoadBatch(1)
    X_val_raw, Y_val, y_val = LoadBatch(2)
    X_test_raw, Y_test, y_test = LoadBatch("test_batch")

    X_train, X_val, X_test = preprocess_data(X_train_raw, X_val_raw, X_test_raw)

    # Define experiment configurations
    configs = [
        {"lam": 0, "eta": 0.1, "label": "lam0_eta01"},
        {"lam": 0, "eta": 0.001, "label": "lam0_eta0001"},
        {"lam": 0.1, "eta": 0.001, "label": "lam01_eta0001"},
        {"lam": 1, "eta": 0.001, "label": "lam1_eta0001"}
    ]

    

    for cfg in configs:
        print(f"Running config: lambda={cfg['lam']}, eta={cfg['eta']}")
        init_net = initialize_network()
        GDparams = {"n_batch": 100, "eta": cfg["eta"], "n_epochs": 40}

        # Train network
        trained_net, loss_hist, cost_hist = MiniBatchGD(
            X_train, Y_train, GDparams, init_net, cfg["lam"],
            X_val=X_val, Y_val=Y_val, track_loss=True
        )

        # Evaluate final test accuracy
        P_test = ApplyNetwork(X_test, trained_net)
        test_acc = ComputeAccuracy(P_test, y_test[0])
        print(f"Test Accuracy for {cfg['label']}: {test_acc:.2f}%")

        # Plot training and validation loss
        plt.figure()
        plt.plot(loss_hist["train"], label="Train Loss")
        plt.plot(loss_hist["val"], label="Val Loss")
        plt.title(f"Loss Curve ({cfg['label']})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"../results/images/loss_curve_{cfg['label']}.png")

        # Plot training and validation Cost
        plt.figure()
        plt.plot(cost_hist["train"], label="Train Cost")
        plt.plot(cost_hist["val"], label="Val Cost")
        plt.title(f"Cost Curve ({cfg['label']})")
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"../results/images/cost_curve_{cfg['label']}.png")

        # Visualize weights
        W = trained_net["W"]
        Ws = W.T.reshape((32, 32, 3, 10), order='F')
        W_img = np.transpose(Ws, (1, 0, 2, 3))

        fig, axs = plt.subplots(1, 10, figsize=(15, 2))
        for i in range(10):
            w_img = W_img[:, :, :, i]
            w_img_norm = (w_img - np.min(w_img)) / (np.max(w_img) - np.min(w_img))
            axs[i].imshow(w_img_norm)
            axs[i].axis('off')
            axs[i].set_title(class_names[i], fontsize=8, fontweight='bold')

        plt.suptitle(f"Weight Visualization ({cfg['label']})")
        plt.savefig(f"../results/images/weights_{cfg['label']}.png")
        plt.close()




def main_use_all_batches():
    # Load all 5 training batches
    X_train, Y_train, y_train = [], [], []
    for i in range(1, 6):
        X, Y, y = LoadBatch(i)
        y=y[0]
        X_train.append(X)
        Y_train.append(Y)
        y_train.append(y)

    X_train = np.concatenate(X_train, axis=1)
    Y_train = np.concatenate(Y_train, axis=1)
    y_train = np.concatenate(y_train)

    # Split off 1000 samples for validation
    X_val, Y_val, y_val = X_train[:, -1000:], Y_train[:, -1000:], y_train[-1000:]
    X_train, Y_train, y_train = X_train[:, :-1000], Y_train[:, :-1000], y_train[:-1000]

    # Normalize
    mean_X = np.mean(X_train, axis=1).reshape(-1, 1)
    std_X = np.std(X_train, axis=1).reshape(-1, 1)
    X_train = (X_train - mean_X) / std_X
    X_val = (X_val - mean_X) / std_X

    # Init network
    network = initialize_network(seed=42)

    n_batch = 100
    eta = 0.0001
    n_epochs = 40
    lam = 0.1
    # Train
    GDparams = {'n_batch': n_batch, 'eta': eta, 'n_epochs': n_epochs}
    trained_net, loss_hist, cost_hist  = MiniBatchGD(X_train, Y_train,GDparams, network, lam,  X_val=X_val, Y_val=Y_val, track_loss=True)

    # Plot training and validation loss
    plot_curve(loss_hist, n_batch, eta, n_epochs, label = "Loss", save_path = "../results/exercise_2/01/loss_curve_ALLBATCHES.png")    
    plot_curve(cost_hist,  n_batch, eta, n_epochs, label = "Coss", save_path = "../results/exercise_2/01/cost_curve_ALLBATCHES.png")
    

    # Visualize weights
    visualize_weights(W=trained_net["W"], n_batch=n_batch, eta=eta, n_epochs=n_epochs, save_path="../results/exercise_2/01/weights_ALLBATCHES.png")

    # Evaluate
    evaluate(trained_net, X_val, Y_val, y_val)

def main_with_horizontal_flipping():
    X_train, Y_train, y_train = LoadBatch(1)

    # Normalize
    mean_X = np.mean(X_train, axis=1).reshape(-1, 1)
    std_X = np.std(X_train, axis=1).reshape(-1, 1)
    X_train = (X_train - mean_X) / std_X

    # Compute flipping indices
    aa = np.int32(np.arange(32)).reshape((32, 1))
    bb = np.int32(np.arange(31, -1, -1)).reshape((32, 1))
    vv = np.tile(32 * aa, (1, 32))
    ind_flip = vv.reshape((32*32, 1)) + np.tile(bb, (32, 1))
    inds_flip = np.vstack((ind_flip, 1024 + ind_flip, 2048 + ind_flip))

    def augment(X):
        flip_mask = np.random.rand(X.shape[1]) < 0.5
        X_aug = X.copy()
        X_aug[:, flip_mask] = X[inds_flip.flatten(), :][:, flip_mask]
        return X_aug

    # Init net
    network = initialize_network(seed=42)

    # Mini-batch training with augmentation
    GDparams = {'n_batch': 100, 'eta': 0.001, 'n_epochs': 40}
    trained_net = MiniBatchGD(X_train, Y_train, GDparams, network, lam=0.01, augment_fn=augment)

    # Evaluate
    _, _, y_val = LoadBatch("Datasets/cifar-10-batches-py/data_batch_2")
    evaluate(trained_net, X_train, Y_train, y_train)



if __name__ == "__main__":
    #exercise_1()
    main_use_all_batches()