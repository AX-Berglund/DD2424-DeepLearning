# import torch
# import numpy as np

# def ComputeGradsWithTorch(X, y, network_params):
    
#     Xt = torch.from_numpy(X)

#     L = len(network_params['W'])

#     # will be computing the gradient w.r.t. these parameters    
#     W = [None] * L
#     b = [None] * L    
#     for i in range(len(network_params['W'])):
#         W[i] = torch.tensor(network_params['W'][i], requires_grad=True)
#         b[i] = torch.tensor(network_params['b'][i], requires_grad=True)        

#     ## give informative names to these torch classes        
#     apply_relu = torch.nn.ReLU()
#     apply_softmax = torch.nn.Softmax(dim=0)

#     #### BEGIN your code ###########################
    
#     # Apply the scoring function corresponding to equations (1-3) in assignment description 
#     # If X is d x n then the final scores torch array should have size 10 x n 

#     # Forward pass
#     s1 = W[0] @ Xt + b[0]  # First layer pre-activation
#     h = apply_relu(s1)  # ReLU activation
#     scores = W[1] @ h + b[1]  # Second layer pre-activation

#     #### END of your code ###########################            

#     # apply SoftMax to each column of scores     
#     P = apply_softmax(scores)
    
#     # compute the loss
#     n = X.shape[1]
#     loss = torch.mean(-torch.log(P[y, np.arange(n)]))
    
#     # compute the backward pass relative to the loss and the named parameters 
#     loss.backward()

#     # extract the computed gradients and make them numpy arrays 
#     grads = {}
#     grads['W'] = [None] * L
#     grads['b'] = [None] * L
#     for i in range(L):
#         grads['W'][i] = W[i].grad.numpy()
#         grads['b'][i] = b[i].grad.numpy()

#     return grads

import torch
import numpy as np

def ComputeGradsWithTorch(X, y, network_params, lam):
    
    Xt = torch.from_numpy(X)

    L = len(network_params['W'])

    # Will be computing the gradient w.r.t. these parameters    
    W = [None] * L
    b = [None] * L    
    for i in range(L):
        W[i] = torch.tensor(network_params['W'][i], requires_grad=True)
        b[i] = torch.tensor(network_params['b'][i], requires_grad=True)        

    # Give informative names to these torch classes        
    apply_relu = torch.nn.ReLU()
    apply_softmax = torch.nn.Softmax(dim=0)

    # Forward pass
    s1 = W[0] @ Xt + b[0]  # First layer pre-activation
    h = apply_relu(s1)  # ReLU activation
    scores = W[1] @ h + b[1]  # Second layer pre-activation

    # Apply SoftMax to each column of scores     
    P = apply_softmax(scores)
    
    # Compute the loss
    n = X.shape[1]
    cross_entropy_loss = torch.mean(-torch.log(P[y, np.arange(n)]))

    # Compute L2 regularization term
    l2_reg = sum(torch.sum(w**2) for w in W)

    # Total cost function (loss + L2 regularization)
    cost = cross_entropy_loss + lam * l2_reg
    
    # Compute gradients
    cost.backward()

    # Extract the computed gradients and make them numpy arrays 
    grads = {'W': [None] * L, 'b': [None] * L}
    for i in range(L):
        grads['W'][i] = W[i].grad.numpy()
        grads['b'][i] = b[i].grad.numpy()

    return grads
