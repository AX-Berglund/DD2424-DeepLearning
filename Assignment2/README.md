# README - Assignment 2 (DD2424)

## Overview
This project involves training and testing a **two-layer neural network** with multiple outputs to classify images from the **CIFAR-10 dataset**. The network is trained using **mini-batch gradient descent** applied to a cost function that computes the **cross-entropy loss** with an additional **L2 regularization** term on the weight matrices. The assignment explores **cyclical learning rates** for improved training efficiency.

## Objectives
1. Implement a two-layer neural network for classification.
2. Utilize **ReLU activation** in the hidden layer.
3. Train the network using **mini-batch gradient descent with cyclic learning rates**.
4. Evaluate different values of the **L2 regularization parameter** (lambda).
5. Compare validation and training performance to optimize hyperparameters.

## Network Architecture
The network consists of:
- **Input layer**: \( d \times 1 \) (image vector from CIFAR-10)
- **Hidden layer**: \( m \times d \) weights and \( m \times 1 \) biases with **ReLU activation**
- **Output layer**: \( K \times m \) weights and \( K \times 1 \) biases with **softmax activation**

The forward pass follows:
\[
s_1 = W_1 x + b_1
\]
\[
h = \max(0, s_1) \quad \text{(ReLU Activation)}
\]
\[
s = W_2 h + b_2
\]
\[
p = \text{softmax}(s)
\]

## Installation & Setup
### Prerequisites
Ensure you have **Python 3.x** and the following libraries installed:
```bash
pip install numpy matplotlib torch
```

### Dataset Setup
Download the CIFAR-10 dataset and extract it into the `Datasets` directory:
```bash
mkdir -p Datasets
cd Datasets
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```
Extract the dataset:
```bash
tar -xvzf cifar-10-python.tar.gz
cd ..
```

## Implementation Details
### 1. Data Preprocessing
- Load CIFAR-10 batch files.
- Normalize pixel values to have **zero mean and unit variance**.

### 2. Model Initialization
- **Weights**: Initialized using **Gaussian distribution**.
- **Biases**: Initialized to **zero**.

### 3. Training with Cyclic Learning Rates
- **Learning rate oscillates between** `η_min = 1e-5` and `η_max = 1e-1`.
- Update rule:
  \[
  \eta_t = \eta_{min} + \frac{t \mod (2n_s)}{n_s} (\eta_{max} - \eta_{min})
  \]
  for increasing and decreasing phase.

- Gradient Descent Update:
  \[
  W_k = W_k - \eta \frac{\partial J}{\partial W_k}
  \]
  \[
  b_k = b_k - \eta \frac{\partial J}{\partial b_k}
  \]

### 4. Gradient Computation
- Implement **backpropagation** for computing gradients.
- Validate gradients using **PyTorch’s autograd**.

### 5. Hyperparameter Search
- Perform **coarse-to-fine grid search** for `λ`.
- Use **random search** over log scale values.

## Running the Code
### Training the Model
Run the following command to train the model:
```bash
python Assignment2.py --train
```

### Testing the Model
To evaluate the trained model:
```bash
python Assignment2.py --test
```

### Hyperparameter Search
To perform hyperparameter tuning:
```bash
python Assignment2.py --search
```

## Results
- **Training loss and validation loss curves** plotted per epoch.
- **Final test accuracy** achieved using optimized hyperparameters.
- **Effect of λ on model performance**.

## Bonus Implementations
- **Data augmentation** (random flipping and translations)
- **Dropout for regularization**
- **Comparison with Adam optimizer**

## References
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Smith, 2015] Cyclical Learning Rates for Training Neural Networks
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

