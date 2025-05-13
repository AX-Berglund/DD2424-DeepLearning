import numpy as np
import torch

def ComputeGradsWithTorch(X, y, h0, RNN):
    """
    Compute gradients using PyTorch's automatic differentiation.
    
    Args:
        X: Input data, shape (d, tau)
        y: Target indices, shape (tau,)
        h0: Initial hidden state, shape (m, 1)
        RNN: Dictionary containing the RNN parameters
    
    Returns:
        grads: Dictionary of gradients for all parameters
    """
    tau = X.shape[1]  # number of time steps

    Xt = torch.from_numpy(X)
    ht = torch.from_numpy(h0)

    torch_network = {}
    for kk in RNN.keys():
        torch_network[kk] = torch.tensor(RNN[kk], requires_grad=True)

    # Give informative names to these torch classes        
    apply_tanh = torch.nn.Tanh()
    apply_softmax = torch.nn.Softmax(dim=0) 
    
    # Create an empty tensor to store the hidden vector at each timestep
    Hs = torch.empty(h0.shape[0], X.shape[1], dtype=torch.float64)
    
    hprev = ht
    for t in range(tau):
        # Equation (1): Compute the hidden scores
        a = torch.matmul(torch_network['W'], hprev) + torch.matmul(torch_network['U'], Xt[:, t:t+1]) + torch_network['b']
        
        # Equation (2): Apply tanh activation
        hnext = apply_tanh(a)
        
        # Store the hidden state
        Hs[:, t] = hnext.squeeze(1)
        
        # Update hprev for the next time step
        hprev = hnext

    # Equation (3): Compute the output scores
    Os = torch.matmul(torch_network['V'], Hs) + torch_network['c']
    
    # Equation (4): Apply softmax
    P = apply_softmax(Os)    
    
    # Compute the loss (cross-entropy)
    loss = torch.mean(-torch.log(P[y, torch.arange(tau)]))
    
    # Compute the backward pass relative to the loss and the named parameters 
    loss.backward()

    # Extract the computed gradients and make them numpy arrays
    grads = {}
    for kk in RNN.keys():
        grads[kk] = torch_network[kk].grad.numpy()

    return grads