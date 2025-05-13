# RNN Text Generation - Harry Potter and the Goblet of Fire

This project implements a character-level Recurrent Neural Network (RNN) to generate text in the style of J.K. Rowling's "Harry Potter and the Goblet of Fire". The implementation follows the requirements of the DD2424 - Assignment 4.

## Project Structure

```
rnn_text_generation/
├── data/
│   └── goblet_book.txt     # The training data
├── src/
│   ├── __init__.py         # Make it a proper package
│   ├── utils.py            # Helper functions
│   ├── data.py             # Data preparation functions
│   ├── model.py            # RNN model definition
│   ├── train.py            # Training loop implementation
│   ├── gradient_check.py   # Gradient checking functions
│   └── torch_grads.py      # PyTorch gradient computation
├── notebook/
│   └── rnn_text_generation.ipynb  # For experimenting and plotting
├── main.py                 # Main script to run training
├── verify_gradients.py     # Script to verify gradient computations
└── README.md               # Project documentation
```

## Features

- Character-level RNN for text generation
- Implementation of forward and backward passes for backpropagation
- Adam optimizer for parameter updates
- Text synthesis with various sampling strategies (standard, temperature, nucleus)
- Gradient checking against numerical gradients and PyTorch implementation
- Visualization of training loss and text samples

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- PyTorch (for gradient checking)

## Getting Started

1. Place the `goblet_book.txt` file in the `data/` directory.

2. Install the required packages:
   ```
   pip install numpy matplotlib torch
   ```

3. Verify the gradient computations:
   ```
   python verify_gradients.py
   ```

4. Run the main script to train the model:
   ```
   python main.py --train
   ```

5. Or run with specific options:
   ```
   python main.py --train --check-gradients --num-updates 10000 --hidden-size 100 --seq-length 25 --learning-rate 0.001 --optimized
   ```
   The `--optimized` flag enables performance optimizations for the forward and backward passes.

6. To compare performance between standard and optimized implementations:
   ```
   python benchmark_performance.py
   ```

6. Alternatively, explore the project through the Jupyter notebook:
   ```
   jupyter notebook notebook/rnn_text_generation.ipynb
   ```

## Implementation Details

### Model Architecture

The vanilla RNN is implemented as described in the assignment, with the following operations:

1. For each time step t:
   - a_t = W·h_(t-1) + U·x_t + b
   - h_t = tanh(a_t)
   - o_t = V·h_t + c
   - p_t = softmax(o_t)

2. The loss is computed as the cross-entropy loss:
   - L = -(1/τ)·∑(y_t^T·log(p_t))

### Optimizer

The Adam optimizer is implemented with the following update steps:

1. m_θ,t' = β1·m_θ,t'-1 + (1-β1)·g_t'
2. v_θ,t' = β2·v_θ,t'-1 + (1-β2)·g_t'^2
3. m̂_θ,t' = m_θ,t'/(1-β1^t')
4. v̂_θ,t' = v_θ,t'/(1-β2^t')
5. θ_t'+1 = θ_t' - η·m̂_θ,t'/(sqrt(v̂_θ,t')+ε)

Default hyperparameters: β1 = 0.9, β2 = 0.999, ε = 1e-8

### Gradient Checking

The implementation includes two methods for gradient checking:

1. **Numerical Gradient Checking**: Computes gradients by perturbing each parameter slightly and comparing the resulting change in loss with the analytically computed gradients.

2. **PyTorch Gradient Checking**: Implements the same RNN model in PyTorch and uses its automatic differentiation engine to compute gradients for comparison. The PyTorch implementation follows the provided assignment template.

### Text Synthesis

The text synthesis function generates new text by sampling the next character from the predicted probability distribution. Three sampling methods are implemented:

1. Standard sampling: Sample directly from the softmax probabilities.
2. Temperature sampling: Apply a temperature parameter to adjust the "peakiness" of the distribution.
3. Nucleus sampling: Only sample from the top k probabilities that sum to a threshold θ.

## Assignment Results

The model was trained on the text of "Harry Potter and the Goblet of Fire" for 100,000 update steps. The training loss decreased from an initial value of ~4.6 to ~1.5 after training.

Text samples from different stages of training show how the model progressively learns to generate more coherent text:
- Initial: Random characters with no structure
- Early (~1,000 iterations): Simple character patterns emerge
- Middle (~10,000 iterations): Word-like structures and some common words
- Later (~30,000 iterations): Coherent phrases and sentences
- Final: Text that resembles the style of the book, with recognizable character names and dialogue

## Additional Improvements

The optional improvements implemented include:

1. Temperature sampling: Applying a temperature parameter to the softmax to control the diversity of generated text.
2. Nucleus sampling: Implementing the method from "The Curious Case of Neural Text Degeneration" paper.
3. Optimized gradient computations for speed.

## Author

This implementation was created for the DD2424 - Assignment 4.py   # Gradient checking functions
├── notebook/
│   └── rnn_text_generation.ipynb  # For experimenting and plotting
├── main.py                 # Main script to run training
└── README.md               # Project documentation
```

## Features

- Character-level RNN for text generation
- Implementation of forward and backward passes for backpropagation
- Adam optimizer for parameter updates
- Text synthesis with various sampling strategies (standard, temperature, nucleus)
- Gradient checking against numerical gradients and PyTorch implementation
- Visualization of training loss and text samples

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- PyTorch (for gradient checking)

## Getting Started

1. Place the `goblet_book.txt` file in the `data/` directory.

2. Install the required packages:
   ```
   pip install numpy matplotlib torch
   ```

3. Run the main script to train the model:
   ```
   python main.py
   ```

4. Alternatively, explore the project through the Jupyter notebook:
   ```
   jupyter notebook notebook/rnn_text_generation.ipynb
   ```

## Implementation Details

### Model Architecture

The vanilla RNN is implemented as described in the assignment, with the following operations:

1. For each time step t:
   - a_t = W·h_(t-1) + U·x_t + b
   - h_t = tanh(a_t)
   - o_t = V·h_t + c
   - p_t = softmax(o_t)

2. The loss is computed as the cross-entropy loss:
   - L = -(1/τ)·∑(y_t^T·log(p_t))

### Optimizer

The Adam optimizer is implemented with the following update steps:

1. m_θ,t' = β1·m_θ,t'-1 + (1-β1)·g_t'
2. v_θ,t' = β2·v_θ,t'-1 + (1-β2)·g_t'^2
3. m̂_θ,t' = m_θ,t'/(1-β1^t')
4. v̂_θ,t' = v_θ,t'/(1-β2^t')
5. θ_t'+1 = θ_t' - η·m̂_θ,t'/(sqrt(v̂_θ,t')+ε)

Default hyperparameters: β1 = 0.9, β2 = 0.999, ε = 1e-8

### Text Synthesis

The text synthesis function generates new text by sampling the next character from the predicted probability distribution. Three sampling methods are implemented:

1. Standard sampling: Sample directly from the softmax probabilities.
2. Temperature sampling: Apply a temperature parameter to adjust the "peakiness" of the distribution.
3. Nucleus sampling: Only sample from the top k probabilities that sum to a threshold θ.

## Assignment Results

The model was trained on the text of "Harry Potter and the Goblet of Fire" for 100,000 update steps. The training loss decreased from an initial value of ~4.6 to ~1.5 after training.

Text samples from different stages of training show how the model progressively learns to generate more coherent text:
- Initial: Random characters with no structure
- Early (~1,000 iterations): Simple character patterns emerge
- Middle (~10,000 iterations): Word-like structures and some common words
- Later (~30,000 iterations): Coherent phrases and sentences
- Final: Text that resembles the style of the book, with recognizable character names and dialogue

## Additional Improvements

The optional improvements implemented include:

1. Temperature sampling: Applying a temperature parameter to the softmax to control the diversity of generated text.
2. Nucleus sampling: Implementing the method from "The Curious Case of Neural Text Degeneration" paper.
3. Optimized gradient computations for speed.

## Author

This implementation was created for the DD2424 - Assignment 4.