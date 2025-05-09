# Assignment 3: Convolutional Neural Network with Initial Patchify Layer

This repository contains the implementation of a three-layer neural network with an initial patchify (convolution) layer for classifying images from the CIFAR-10 dataset.

## Project Structure

```
assignment3/
│
├── data/                       # Data directory
│   └── debug_info.npz         # Debug data file
│
├── src/                        # Source code
│   ├── __init__.py            # Makes the directory a package
│   ├── convolution.py         # Convolution implementations (Exercise 1)
│   ├── gradients.py           # Gradient computation (Exercise 2)
│   ├── network.py             # Neural network implementation
│   ├── training.py            # Training functions
│   └── utils.py               # Utility functions
│
├── main.py                     # Main script to run experiments
└── README.md                   # Project documentation
```

## Quick Start


### Setup

1. Ensure you have Python 3.7+ installed
2. Install the required packages:
   ```
   pip install numpy matplotlib
   ```

3. Download the CIFAR-10 dataset and extract it into the `data` directory:
   ```bash
   mkdir -p data
   cd data
   curl -O https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
   ```
4. Extract the dataset:
   ```bash
   tar -xvzf cifar-10-python.tar.gz
   cd ..
   ```

## Exercises

### Exercise 1: Efficient Convolution Implementation

Implements and compares three different methods for performing convolution:
- Dot product approach
- Matrix multiplication approach
- Einstein summation approach

Run with:
```bash
python main.py --exercise 1
```

### Exercise 2: Gradient Computation

Implements forward and backward passes for the network, with gradient verification.

Run with:
```bash
python main.py --exercise 2
```

### Exercise 3: Training Small Networks with Cyclic Learning Rates

Trains and compares four different network architectures:
- Architecture 1: f = 2, nf = 3, nh = 50
- Architecture 2: f = 4, nf = 10, nh = 50
- Architecture 3: f = 8, nf = 40, nh = 50
- Architecture 4: f = 16, nf = 160, nh = 50

Run with:
```bash
python main.py --exercise 3 --save_plots
```

### Exercise 4: Larger Networks and Label Smoothing

Trains a larger network (f = 4, nf = 40, nh = 300) with and without label smoothing and compares the results.

Run with:
```bash
python main.py --exercise 4 --save_plots
```

## All Available Run Commands

```bash
# Run specific architectures
python exercise3.py --architecture 1  # f=2, nf=3, nh=50
python exercise3.py --architecture 2  # f=4, nf=10, nh=50
python exercise3.py --architecture 3  # f=8, nf=40, nh=50
python exercise3.py --architecture 4  # f=16, nf=160, nh=50
python exercise3.py --architecture 5  # f=4, nf=40, nh=300 (with label smoothing comparison)

# Run longer training with increasing step sizes
python exercise3.py --longer

# Use label smoothing
python exercise3.py --label_smoothing

# Specify number of training samples
python exercise3.py --n_train 49000

# Specify custom data path
python exercise3.py --data_path /path/to/cifar-10-batches-py

# Specify number of threads
python exercise3.py --num_threads 4

# Combine options
python exercise3.py --architecture 5 --n_train 49000 --num_threads 4
python exercise3.py --architecture 2 --longer --label_smoothing

# Show help
python exercise3.py --help
```

## Network Architecture

The network consists of three layers:
1. Convolutional (patchify) layer with multiple filters
2. Hidden layer with ReLU activation
3. Output layer with softmax activation
