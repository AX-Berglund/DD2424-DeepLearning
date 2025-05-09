# DD2424 – Assignment 3

In this assignment you will train and test a three-layer network with multiple outputs to classify images from the CIFAR-10 dataset. The first layer will be a convolution layer applied with a stride equal to the width of the filter. In the parlance of computer vision this corresponds to a **patchify layer** (see Figure 1). This is the first layer that is applied in the Vision Transformer and other architectures such as MLPMixer and ConvMixer.

You will train the network using **mini-batch gradient descent** applied to a cost function computing:

- the **cross-entropy loss** of the classifier applied to the labeled training data
- and an **L2 regularization term** on the weight matrix.

---

## Figure 1: Patchify

To patchify an input image, split it into a regular grid of non-overlapping sub-regions. For Vision Transformers the pixel data in each patch is flattened into a vector, transformed with an affine transformation, and this output vector becomes an input to a Transformer network. The patchify operation is just a convolution applied with stride equal to the width of the filter.

---

## Code Structure

The overall structure of your code for this assignment should mimic that from the previous assignments. You will have slightly different parameters and will have to change the functions that:

1. Evaluate the network (the forward pass), and
2. Compute the gradients (the backward pass).

There is some work to do to get an efficient implementation of the first convolutional layer to allow reasonable training times without a GPU — but the reward will be improved performance from assignment 2.

---

## Background 1: Network with an Initial Patchify Layer

Each input image $X$ in its original form is a 3D array of size $32 \times 32 \times 3$.

The network function you will apply to the input $X$ is:

$$
H_i = \max(0, X * F_i) \quad \text{for } i = 1, \dots, n_f \tag{1}
$$

$$
h = 
\begin{pmatrix}
\text{vec}(H_1) \\
\text{vec}(H_2) \\
\vdots \\
\text{vec}(H_{n_f})
\end{pmatrix} \tag{2}
$$

$$
x_1 = \max(0, W_1 h + b_1) \tag{3}
$$

$$
s = W_2 x_1 + b_2 \tag{4}
$$

$$
p = \text{SoftMax}(s) \tag{5}
$$

Where:

- $X$: input image of size $32 \times 32 \times 3$
- $F_i$: filter of size $f \times f \times 3$, applied with stride $f$, no zero-padding
- $f \in \{2, 4, 8, 16\}$
- $H_i$: output of each convolution, size $\frac{32}{f} \times \frac{32}{f} \times 1$
- $\text{vec}(\cdot)$: flatten operation
- $h$: concatenated vector of all $H_i$, of size $n_f n_p \times 1$, where $n_p = (32/f)^2$
- $W_1$: size $d \times d_0$, with $d_0 = n_f n_p$
- $W_2$: size $K \times d$
- SoftMax:
  $$
  \text{SoftMax}(s) = \frac{\exp(s)}{1^T \exp(s)} \tag{6}
  $$
- Predicted class:
  $$
  k^* = \arg \max_{1 \leq k \leq K} \{p_1, \dots, p_K\} \tag{7}
  $$

### Note

The operator `vec()` flattens a matrix row-by-row:

$$
H = 
\begin{pmatrix}
H_{11} & H_{12} \\
H_{21} & H_{22}
\end{pmatrix}
\Rightarrow
\text{vec}(H) = 
\begin{pmatrix}
H_{11} \\
H_{12} \\
H_{21} \\
H_{22}
\end{pmatrix}
$$

Bias terms are omitted for simplicity.

---

## Background 2: Writing the Convolution as a Matrix Multiplication

To make back-propagation efficient (especially on CPU), we write convolutions as matrix multiplications.

Example:

Let $X$ be a $4 \times 4$ input matrix, and $F$ a $2 \times 2$ filter.

$$
X = 
\begin{pmatrix}
X_{11} & X_{12} & X_{13} & X_{14} \\
X_{21} & X_{22} & X_{23} & X_{24} \\
X_{31} & X_{32} & X_{33} & X_{34} \\
X_{41} & X_{42} & X_{43} & X_{44}
\end{pmatrix},
\quad
F =
\begin{pmatrix}
F_{11} & F_{12} \\
F_{21} & F_{22}
\end{pmatrix} \tag{8}
$$

With stride 2 and no padding, the output is $2 \times 2$:

$$
H = X * F \Rightarrow h = M_X \cdot \text{vec}(F) \tag{9}
$$

Matrix $M_X$:

$$
M_X = 
\begin{pmatrix}
X_{11} & X_{12} & X_{21} & X_{22} \\
X_{13} & X_{14} & X_{23} & X_{24} \\
X_{31} & X_{32} & X_{41} & X_{42} \\
X_{33} & X_{34} & X_{43} & X_{44}
\end{pmatrix} \tag{10}
$$

---

### Multiple Channels

Let $X \in \mathbb{R}^{4 \times 4 \times 2}$, $F \in \mathbb{R}^{2 \times 2 \times 2}$.

Then $M_X$ becomes:

$$
M_X = 
\begin{pmatrix}
X_{111} & X_{121} & X_{211} & X_{221} & X_{112} & X_{122} & X_{212} & X_{222} \\
X_{131} & X_{141} & X_{231} & X_{241} & X_{132} & X_{142} & X_{232} & X_{242} \\
X_{311} & X_{321} & X_{411} & X_{421} & X_{312} & X_{322} & X_{412} & X_{422} \\
X_{331} & X_{341} & X_{431} & X_{441} & X_{332} & X_{342} & X_{432} & X_{442}
\end{pmatrix} \tag{12}
$$

$$
\text{vec}(F) = 
\begin{pmatrix}
F_{111}, F_{121}, F_{211}, F_{221}, F_{112}, F_{122}, F_{212}, F_{222}
\end{pmatrix}^T \tag{13}
$$

---

### Propagating the Gradient to $F$

Gradient w.r.t. the filter:

$$
\frac{\partial L}{\partial \text{vec}(F)} = M_X^T g \tag{14}
$$

For a batch:

$$
\frac{\partial L}{\partial \text{vec}(F)} = \frac{1}{|B|} \sum_{(X, y) \in B} M_X^T g_y \tag{15}
$$

---

## Background 3: Multiple Convolution Filters

Apply multiple filters $F_1, F_2, F_3$:

$$
F_{\text{all}} = [\text{vec}(F_1), \text{vec}(F_2), \text{vec}(F_3)] \tag{17}
$$

Apply to patches:

$$
H = M_X F_{\text{all}} \tag{16}
$$

Gradient:

$$
\frac{\partial L}{\partial F_{\text{all}}} = M_X^T G \tag{18}
$$

Where:

$$
G = [g_1, g_2, g_3] \tag{19}
$$

---

## Background 4: Backpropagation Equations

Let:

$$
L(B, \Theta) = \frac{1}{|B|} \sum_{(x, y) \in B} \ell_{\text{cross}}(y, f_{\text{network}}(X, \Theta)) \tag{20}
$$

Gradient w.r.t. $F_{\text{all}}$:

$$
\frac{\partial L(B)}{\partial F_{\text{all}}} = \frac{1}{|B|} \sum_{(X, y) \in B} M_X^T G_y \tag{21}
$$

In code:

- $M$: tensor of shape $n_p \times 3f^2 \times n$
- $G$: tensor of shape $n_p \times n_f \times n$

Then:

$$
F^{\text{grad}}_{\text{all}} = \frac{1}{n} \sum_{i=1}^{n} M(:, :, i)^T G(:, :, i) \tag{22}
$$

---

## Background 5: Label Smoothing (Regularization)

Instead of one-hot label vector $y$, use smoothed version:

$$
y_{\text{smooth}, i} =
\begin{cases}
1 - \epsilon & \text{if } i = y \\
\frac{\epsilon}{K - 1} & \text{otherwise}
\end{cases} \tag{23}
$$

Typical $\epsilon = 0.1$

Update backward pass:

$$
-(y - p) \Rightarrow -(y_{\text{smooth}} - p) \tag{24}
$$

---

## Background 6: Cyclical Learning Rates with Increasing Steps

In Assignment 2 you used cyclical learning rates. Now upgrade by **doubling the step size** per cycle:

$$
n_{i+1, s} = 2 n_{i, s} \tag{25}
$$

This approximates **cosine with warm restarts** ([Loshchilov and Hutter, 2017]).

---

## ✅ Exercise 1 Write code to implement the convolution efficiently



### Step 1: Dot-product version

Load debugging data:

* `debug_file = 'debug_conv_info.npz'`
* `load_data = np.load(debug_file)`
* `X = load_data['X']` (shape: 3072 × n with n = 5)
* `Fs = load_data['Fs']` (shape: f × f × 3 × nf, with f = 4, nf = 2)

Reshape and transpose:

* `X_ims = np.transpose(X.reshape((32, 32, 3, n), order='F'), (1, 0, 2, 3))`

Loop through each image, each filter, and each f × f × 3 sub-patch. Use `np.multiply` and `np.sum` for dot products.

Compare your output to:

* `load_data['conv_outputs']`

---

### Step 2: Matrix Multiplication Version

Create MX of shape (n\_p, f \* f \* 3, n):

* `MX[l, :, i] = X_patch.reshape((1, f * f * 3), order='C')`

Flatten the filters:

* `Fs_flat = Fs.reshape((f * f * 3, nf), order='C')`

Matrix multiplication loop:

* `conv_outputs_mat[:, :, i] = np.matmul(MX[:, :, i], Fs_flat)`

Compare with:

* `conv_outputs_flat = conv_outputs.reshape((n_p, nf, n), order='C')`

---

### Step 3: Use Einstein Summation

Replace loop with einsum:

* `conv_outputs_mat = np.einsum('ijn, jl -> iln', MX, Fs_flat, optimize=True)`

Check it matches previous output.

---

## ✅ Exercise 2: Compute Gradients

### Forward Pass

Use:

* `conv_flat = np.fmax(conv_outputs_mat.reshape((n_p * nf, n), order='C'), 0)`

Parameters provided:

* `W1 = load_data['W1']` (shape: nh × (n\_p \* nf))
* `W2 = load_data['W2']` (shape: 10 × nh)
* `b1 = load_data['b1']` (shape: nh × 1)
* `b2 = load_data['b2']` (shape: 10 × 1)

Forward intermediates:

* `conv_flat = load_data['conv_flat']`
* `X1 = load_data['X1']`
* `P = load_data['P']`

---

### Backward Pass

Target labels:

* `Y = load_data['Y']` (shape: 10 × n)

After computing G\_batch:

* `GG = G_batch.reshape((n_p, nf, n), order='C')`

Compute gradients using:

* `MXt = np.transpose(MX, (1, 0, 2))`
* `grad_Fs_flat = np.einsum('ijn, jln -> il', MXt, GG, optimize=True)`

Compare with:

* `load_data['grad_Fs_flat']`

---

## ✅ Exercise 3: Train Small Networks with Cyclic Learning Rates

Initial setup:

* f = 4, nf = 10, nh = 50
* 3 cycles, step = 800
* η\_min = 1e-5, η\_max = 1e-1
* Batch size = 100
* L2 regularization λ = 0.003

Expected result: \~57.61% test accuracy under 50s (M1 MacBook).

---

### Architecture Comparison

Train and compare:

* Architecture 1: f = 2, nf = 3, nh = 50
* Architecture 2: f = 4, nf = 10, nh = 50
* Architecture 3: f = 8, nf = 40, nh = 50
* Architecture 4: f = 16, nf = 160, nh = 50

Keep output size of convolutional layer constant.

Create bar charts for:

* Final test accuracy
* Training time

---

### Train Longer

Use increasing step sizes:

* Start with step = 800
* 3 cycles

Train:

* Architecture 2
* Architecture 3

Then test with wider architecture:

* Architecture 2, but nf = 40

Compare plots and results.

---

## ✅ Exercise 4: Larger Networks and Label Smoothing

Train Architecture 5:

* f = 4, nf = 40, nh = 300
* 4 cycles, step\_1 = 800
* L2 λ = 0.0025

Compare training/test loss:

* Without label smoothing
* With label smoothing (ε = 0.1)

Comment on differences in overfitting.

---

## Final Deliverables

Submit to Canvas:

1. Code (in one file)
2. PDF report with:

   * Gradient check evidence and training time (Exercise 3)
   * Bar charts: final accuracy + training time (4 architectures)
   * Loss curves for longer training
   * Loss curves with/without label smoothing (Exercise 4)
   * Thoughts on further experiments to improve accuracy

---

## ✅ Exercise 5 (Optional Bonus)

### 5.1: Improve Performance

Try:

* Make the network wider
* Use data augmentation
* Balance L2 and label smoothing
* Decay η\_max over cycles
* Concatenate multi-size filters

Target scores:

* ≥ 68% → +1 bonus point
* ≥ 70% → +2 bonus points

Up to 4 points for improvements, based on best test accuracy.

---

### 5.2: Compare to PyTorch

Use `torch.nn.functional.conv2d` and auto-diff. Compare training time vs. your implementation on CPU for various:

* Filter sizes
* Number of filters

Submit:

1. Code
2. PDF with:

   * Summary of best accuracy and what helped most (5.1)
   * Speed comparisons and conclusions (5.2)

---

## References

* Loshchilov and Hutter (2017): SGDR – Stochastic Gradient Descent with Warm Restarts (ICLR)
* Tan and Le (2019): EfficientNet – Rethinking model scaling for CNNs (ICML)


