
import numpy as np

class RNN:
    def __init__(self, K, m, rng):
        """
        Initialize a vanilla RNN model.
        
        Args:
            K: Input/output dimension (number of unique characters)
            m: Hidden state dimension
            rng: Random number generator
        """
        # Initialize model parameters
        self.params = {
            'b': np.zeros((m, 1)),                # Hidden bias
            'c': np.zeros((K, 1)),                # Output bias
            'U': (1/np.sqrt(2*K)) * rng.standard_normal(size=(m, K)),  # Input-to-hidden weights
            'W': (1/np.sqrt(2*m)) * rng.standard_normal(size=(m, m)),  # Hidden-to-hidden weights
            'V': (1/np.sqrt(m)) * rng.standard_normal(size=(K, m))     # Hidden-to-output weights
        }
        
        # Store dimensions
        self.K = K
        self.m = m
        
        # Initialize Adam optimizer parameters
        self.optimizer = {
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
            'm': {k: np.zeros_like(v) for k, v in self.params.items()},
            'v': {k: np.zeros_like(v) for k, v in self.params.items()},
            't': 0
        }
    
    def forward(self, X, h_prev):
        """
        Forward pass through the RNN.
        
        Args:
            X: Input data, shape (K, seq_length)
            h_prev: Previous hidden state, shape (m, 1)
        
        Returns:
            P: Output probabilities, shape (K, seq_length)
            h: Hidden states, shape (m, seq_length+1)
            a: Pre-activation hidden states, shape (m, seq_length)
            o: Pre-softmax outputs, shape (K, seq_length)
        """
        seq_length = X.shape[1]
        
        # Initialize arrays to store intermediate values
        a = np.zeros((self.m, seq_length))
        h = np.zeros((self.m, seq_length + 1))
        o = np.zeros((self.K, seq_length))
        P = np.zeros((self.K, seq_length))
        
        # Set the initial hidden state
        h[:, 0:1] = h_prev
        
        # Forward pass through time
        for t in range(seq_length):
            # Input to the hidden layer
            a[:, t:t+1] = np.dot(self.params['W'], h[:, t:t+1]) + np.dot(self.params['U'], X[:, t:t+1]) + self.params['b']
            
            # Hidden state
            h[:, t+1:t+2] = np.tanh(a[:, t:t+1])
            
            # Output layer
            o[:, t:t+1] = np.dot(self.params['V'], h[:, t+1:t+2]) + self.params['c']
            
            # Softmax probabilities
            P[:, t:t+1] = np.exp(o[:, t:t+1]) / np.sum(np.exp(o[:, t:t+1]))
        
        return P, h, a, o
    
    def backward(self, X, Y, P, h, a):
        """
        Backward pass through the RNN.
        
        Args:
            X: Input data, shape (K, seq_length)
            Y: Target data, shape (K, seq_length)
            P: Output probabilities from forward pass, shape (K, seq_length)
            h: Hidden states from forward pass, shape (m, seq_length+1)
            a: Pre-activation hidden states from forward pass, shape (m, seq_length)
        
        Returns:
            grads: Dictionary of gradients for all parameters
        """
        seq_length = X.shape[1]
        
        # Initialize gradients
        dU = np.zeros_like(self.params['U'])
        dW = np.zeros_like(self.params['W'])
        dV = np.zeros_like(self.params['V'])
        db = np.zeros_like(self.params['b'])
        dc = np.zeros_like(self.params['c'])
        
        # Initialize gradient of hidden state
        dh_next = np.zeros((self.m, 1))
        
        # Backward pass through time
        for t in reversed(range(seq_length)):
            # Gradient of the output
            do = P[:, t:t+1] - Y[:, t:t+1]
            
            # Gradient of V and c
            dV += np.dot(do, h[:, t+1:t+2].T)
            dc += do
            
            # Gradient of hidden state
            dh = np.dot(self.params['V'].T, do) + dh_next
            
            # Gradient of tanh
            da = (1 - np.square(h[:, t+1:t+2])) * dh
            
            # Gradient of W, U, and b
            db += da
            dW += np.dot(da, h[:, t:t+1].T)
            dU += np.dot(da, X[:, t:t+1].T)
            
            # Gradient for next iteration
            dh_next = np.dot(self.params['W'].T, da)
        
        # Clip gradients to prevent exploding gradients
        for grad in [dU, dW, dV]:
            np.clip(grad, -5, 5, out=grad)
        
        # Store gradients in a dictionary
        grads = {'U': dU, 'W': dW, 'V': dV, 'b': db, 'c': dc}
        
        return grads
    
    def adam_update(self, grads, eta=0.001):
        """
        Update parameters using Adam optimizer.
        
        Args:
            grads: Dictionary of gradients
            eta: Learning rate
        """
        # Increment time step
        self.optimizer['t'] += 1
        t = self.optimizer['t']
        
        # Update for each parameter
        for key in self.params.keys():
            # Update biased first moment estimate
            self.optimizer['m'][key] = self.optimizer['beta1'] * self.optimizer['m'][key] + \
                                      (1 - self.optimizer['beta1']) * grads[key]
            
            # Update biased second moment estimate
            self.optimizer['v'][key] = self.optimizer['beta2'] * self.optimizer['v'][key] + \
                                      (1 - self.optimizer['beta2']) * (grads[key]**2)
            
            # Bias correction
            m_hat = self.optimizer['m'][key] / (1 - self.optimizer['beta1']**t)
            v_hat = self.optimizer['v'][key] / (1 - self.optimizer['beta2']**t)
            
            # Update parameters
            self.params[key] -= eta * m_hat / (np.sqrt(v_hat) + self.optimizer['epsilon'])
    
    def synthesize_text(self, h0, x0, n, ind_to_char, char_to_ind, sampling_strategy='standard', temperature=1.0, theta=0.9, rng=None):
        """
        Synthesize text from the RNN model.
        
        Args:
            h0: Initial hidden state, shape (m, 1)
            x0: Initial input character (one-hot encoded), shape (K, 1)
            n: Number of characters to generate
            ind_to_char: Dictionary mapping indices to characters
            char_to_ind: Dictionary mapping characters to indices
            sampling_strategy: Method for sampling ('standard', 'temperature', or 'nucleus')
            temperature: Temperature parameter for temperature sampling
            theta: Threshold for nucleus sampling
            rng: Random number generator
        
        Returns:
            generated_text: String of generated characters
            Y: One-hot encoding of generated characters, shape (K, n)
        """
        # Initialize arrays
        Y = np.zeros((self.K, n))
        x = x0
        h = h0
        
        # Generate sequence
        for t in range(n):
            # Forward pass for one step
            a = np.dot(self.params['W'], h) + np.dot(self.params['U'], x) + self.params['b']
            h = np.tanh(a)
            o = np.dot(self.params['V'], h) + self.params['c']
            p = np.exp(o) / np.sum(np.exp(o))
            
            # Apply sampling strategy
            if sampling_strategy == 'temperature':
                from src.utils import sample_with_temperature
                p = sample_with_temperature(p, temperature)
            elif sampling_strategy == 'nucleus':
                from src.utils import nucleus_sampling
                p = nucleus_sampling(p, theta)
            
            # Sample the next character
            from src.utils import sample_next_char
            idx = sample_next_char(p, rng)
            
            # Store the generated character
            Y[idx, t] = 1
            
            # Use the generated character as the next input
            x = np.zeros((self.K, 1))
            x[idx, 0] = 1
        
        # Convert one-hot encoding to text
        indices = np.argmax(Y, axis=0)
        generated_text = ''.join([ind_to_char[idx] for idx in indices])
        
        return generated_text, Y