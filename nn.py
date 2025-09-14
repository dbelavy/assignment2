import numpy as np
import autograd.numpy as anp
from autograd import grad


class LinearClassifier:
    """
    Linear classifier: y = Wx + b
    Network structure: [784, 10]
    """
    def __init__(self, n_input, n_output):
        """
        Initialize linear classifier parameters
        Args:
            n_input: Number of input features (784 for Fashion-MNIST)
            n_output: Number of output classes (10 for Fashion-MNIST)
        """
        self.n_input = n_input
        self.n_output = n_output
        
        # Initialize weights and biases with small random values
        self.W = np.random.randn(n_input, n_output) * 0.01
        self.b = np.zeros(n_output)
        
    def forward(self, X):
        """
        Forward pass: compute predictions
        Args:
            X: Input data of shape (batch_size, n_input)
        Returns:
            Output logits of shape (batch_size, n_output)
        """
        # TODO: Implement linear transformation y = Wx + b

        # so the X matrix comes in at batch rows and 784 columns
        #print("Forward pass")
        #print(f"Shape of X: {X.shape}")
        #print(f"Shape of W: {self.W.shape}")
        #print(f"Shape of b: {self.b.shape}")
        #print(f"Let's multiply matrices...!!!")
        # Z = anp.matmul(X, self.W) + self.b
        Z = X @ self.W + self.b

        #print(f"Shape of Z: {Z.shape}")
        #print(f"Z:\n{Z}")
        return Z # Z is the LOGITS
        # remember that LOGIT is the inverse function of the SIGMOID function
        
        # This is a linear classifier that maps input features directly to output logits.
        # You need to:
        # 1. Multiply input X with weight matrix W: X @ W
        #    - X has shape (batch_size, n_input) 
        #    - W has shape (n_input, n_output)
        #    - Result has shape (batch_size, n_output)
        # 2. Add bias term b to each sample
        #    - b has shape (n_output,)
        #    - Broadcasting will handle adding bias to each batch sample
        #
        # Mathematical formula: output = X @ W + b
        # Where @ is matrix multiplication
        #
        # Example:
        # If X is [batch_size=2, n_input=784] and W is [784, 10], b is [10]
        # Then X @ W gives [2, 10] and adding b gives final [2, 10]

    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + anp.exp(-z))
    
    def predict(self, X):
        """
        Make predictions using the model
        Args:
            X: Input data of shape (batch_size, n_input)
        Returns:
            Predicted class labels
        """
        logits = self.forward(X)
        probs = self.sigmoid(logits)
        return np.argmax(probs, axis=1)
    
    def get_params(self):
        """Get model parameters as a flat array"""
        # print("Getting model parameters")
        params = anp.concatenate([self.W.flatten(), self.b.flatten()])
        # print(f"Shape of params: {params.shape}")
        return params
    
    def set_params(self, params):
        """Set model parameters from a flat array"""
        W_size = self.n_input * self.n_output
        self.W = params[:W_size].reshape(self.n_input, self.n_output)
        self.b = params[W_size:]


class TwoLayerMLP:
    """
    Two-layer Multi-Layer Perceptron with one hidden layer
    Network structure: [784, 30, 10]
    """
    def __init__(self, n_input, n_hidden, n_output):
        """
        Initialize MLP parameters
        Args:
            n_input: Number of input features (784)
            n_hidden: Number of hidden units (30)
            n_output: Number of output classes (10)
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        
        # Initialize weights and biases for both layers
        # First layer: input to hidden
        self.W1 = np.random.randn(n_input, n_hidden) * 0.01
        self.b1 = np.zeros(n_hidden)
        
        # Second layer: hidden to output
        self.W2 = np.random.randn(n_hidden, n_output) * 0.01
        self.b2 = np.zeros(n_output)
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + anp.exp(-z))
    
    def forward(self, X):
        """
        Forward pass through the network
        Args:
            X: Input data of shape (batch_size, n_input)
        Returns:
            Output logits of shape (batch_size, n_output)
        """
        # TODO: Implement the forward pass for a two-layer neural network
        # 
        # This network has the structure: Input -> Hidden Layer -> Output Layer
        # Network topology: [784] -> [30] -> [10] (for Fashion-MNIST)
        
        # First layer: linear transformation + sigmoid activation
        # TODO: Implement z1 = X @ W1 + b1, then h1 = sigmoid(z1)
        # layer 1
        z1 = anp.matmul(X, self.W1) + self.b1
        # print(f"Shape of z1: {z1.shape}")
        h1 = self.sigmoid(z1)
        # print(f"Shape of h1 {h1.shape}")
        #layer 2
        z2 = anp.matmul(h1, self.W2) + self.b2
        # print(f"Shape of z2 {z2.shape}")
        # NOTE - not applying second sigmoid function. This will be done later.
        return z2  # Z is the LOGITS
        # 
        # Step-by-step:
        # 1. Linear transformation: z1 = X @ W1 + b1
        #    - X has shape (batch_size, n_input) = (batch_size, 784)
        #    - W1 has shape (n_input, n_hidden) = (784, 30)
        #    - b1 has shape (n_hidden,) = (30,)
        #    - z1 will have shape (batch_size, n_hidden) = (batch_size, 30)
        # 2. Apply sigmoid activation: h1 = sigmoid(z1)
        #    - h1 is the output of the hidden layer, same shape as z1
        #
        # Variables to define:
        # z1 = ...  # Linear transformation result
        # h1 = ...  # Activated hidden layer output

        # Second layer: linear transformation (no activation for output logits)
        # TODO: Implement z2 = h1 @ W2 + b2
        #
        # Step-by-step:
        # 1. Linear transformation: z2 = h1 @ W2 + b2
        #    - h1 has shape (batch_size, n_hidden) = (batch_size, 30)
        #    - W2 has shape (n_hidden, n_output) = (30, 10)
        #    - b2 has shape (n_output,) = (10,)
        #    - z2 will have shape (batch_size, n_output) = (batch_size, 10)
        # 2. z2 represents the final logits (no activation applied here)
        #    - Sigmoid activation will be applied later in predict() or loss function
        #
        # Variable to define:
        # z2 = ...  # Final output logits
    
    def predict(self, X):
        """
        Make predictions using the model
        Args:
            X: Input data of shape (batch_size, n_input)
        Returns:
            Predicted class labels
        """
        logits = self.forward(X)
        probs = self.sigmoid(logits)
        return np.argmax(probs, axis=1)
    
    def get_params(self):
        """Get model parameters as a flat array"""
        # TODO: Concatenate all model parameters into a single flat array
        #
        # This function is needed for automatic differentiation (autograd).
        # The grad() function requires all parameters to be in a single array.
        #

        params = anp.concatenate([self.W1.flatten(), self.b1.flatten(), self.W2.flatten(), self.b2.flatten()])
        # print(f"Shape of params: {params.shape}")
        # this should be a 23860 long array
        return params

        # You need to flatten and concatenate ALL parameters in the correct order:
        # 1. W1: weight matrix of first layer (shape: n_input × n_hidden)
        # 2. b1: bias vector of first layer (shape: n_hidden)
        # 3. W2: weight matrix of second layer (shape: n_hidden × n_output)  
        # 4. b2: bias vector of second layer (shape: n_output)

    
    def set_params(self, params):
        """Set model parameters from a flat array"""
        W1_size = self.n_input * self.n_hidden
        b1_size = self.n_hidden
        W2_size = self.n_hidden * self.n_output
        b2_size = self.n_output
        
        idx = 0
        self.W1 = params[idx:idx + W1_size].reshape(self.n_input, self.n_hidden)
        idx += W1_size
        
        self.b1 = params[idx:idx + b1_size]
        idx += b1_size
        
        self.W2 = params[idx:idx + W2_size].reshape(self.n_hidden, self.n_output)
        idx += W2_size
        
        self.b2 = params[idx:idx + b2_size]


class ThreeLayerMLP:
    """
    Three-layer Multi-Layer Perceptron with two hidden layers
    Network structure: [784, 30, 30, 10] (default)
    """
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output, use_residual=False):
        """
        Initialize MLP parameters
        Args:
            n_input: Number of input features (784)
            n_hidden1: Number of first hidden layer units (30)
            n_hidden2: Number of second hidden layer units (30)
            n_output: Number of output classes (10)
            use_residual: Whether to use residual connections
        """
        self.n_input = n_input
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_output = n_output
        self.use_residual = use_residual
        
        # Check if residual connection is possible (hidden layers must have same size)
        if use_residual and n_hidden1 != n_hidden2:
            raise ValueError("For residual connections, hidden layers must have the same size")
        
        # Initialize weights and biases for all three layers
        # First layer: input to hidden1
        self.W1 = np.random.randn(n_input, n_hidden1) * 0.01
        self.b1 = np.zeros(n_hidden1)
        
        # Second layer: hidden1 to hidden2
        self.W2 = np.random.randn(n_hidden1, n_hidden2) * 0.01
        self.b2 = np.zeros(n_hidden2)
        
        # Third layer: hidden2 to output
        self.W3 = np.random.randn(n_hidden2, n_output) * 0.01
        self.b3 = np.zeros(n_output)
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + anp.exp(-z))
    
    def relu(self, z):
        """ReLU activation function"""
        return anp.maximum(0, z)
    
    def forward(self, X):
        """
        Forward pass through the network
        Args:
            X: Input data of shape (batch_size, n_input)
        Returns:
            Output logits of shape (batch_size, n_output)
        """
        # layer 1
        z1 = anp.matmul(X, self.W1) + self.b1
        # h1 = self.relu(z1)
        h1 = self.sigmoid(z1)
        #layer 2
        z2 = anp.matmul(h1, self.W2) + self.b2
        # h2 = self.relu(z2)
        h2 = self.sigmoid(z2)
        if self.use_residual:
            h2 = h2 + h1

            # raise NotImplementedError  # Residual connection: add previous layer output
        else:
            pass
        # output layer
        z3 = anp.matmul(h2, self.W3) + self.b3
        # no sigmoid
        return z3  # Z is the LOGITS

        # TODO: Implement the forward pass for a three-layer neural network
        # 
        # This network has the structure: Input -> Hidden1 -> Hidden2 -> Output
        # Network topology: [784] -> [30] -> [30] -> [10] (for Fashion-MNIST)
        # With optional residual connections between hidden layers
        
        # First layer: linear transformation + sigmoid activation
        # TODO: Implement z1 = X @ W1 + b1, then h1 = sigmoid(z1)
        # 
        # Step-by-step:
        # 1. Linear transformation: z1 = X @ W1 + b1
        #    - X has shape (batch_size, n_input) = (batch_size, 784)
        #    - W1 has shape (n_input, n_hidden1) = (784, 30)
        #    - b1 has shape (n_hidden1,) = (30,)
        #    - z1 will have shape (batch_size, n_hidden1) = (batch_size, 30)
        # 2. Apply sigmoid activation: h1 = sigmoid(z1)
        
        raise NotImplementedError
        
        # Second layer: linear transformation + sigmoid activation
        # TODO: Implement z2 = h1 @ W2 + b2, then h2_raw = sigmoid(z2)
        #
        # Step-by-step:
        # 1. Linear transformation: z2 = h1 @ W2 + b2
        #    - h1 has shape (batch_size, n_hidden1) = (batch_size, 30)
        #    - W2 has shape (n_hidden1, n_hidden2) = (30, 30)
        #    - b2 has shape (n_hidden2,) = (30,)
        #    - z2 will have shape (batch_size, n_hidden2) = (batch_size, 30)
        # 2. Apply sigmoid activation: h2_raw = sigmoid(z2)
        
        raise NotImplementedError
        
        # Apply residual connection if enabled
        # TODO: Implement residual connection logic
        #
        # Residual connections help with gradient flow in deep networks.
        # The idea is to add the input of a layer to its output: h2 = h2_raw + h1
        # This requires h1 and h2_raw to have the same shape (n_hidden1 == n_hidden2)
        #
        # If residual connections are enabled (self.use_residual == True):
        #   h2 = h2_raw + h1  # Add skip connection from previous layer
        # Else:
        #   h2 = h2_raw       # Use normal activation without skip connection
        
        if self.use_residual:
            raise NotImplementedError  # Residual connection: add previous layer output
        else:
            raise NotImplementedError
        
        # Third layer: linear transformation (output layer)
        # TODO: Implement z3 = h2 @ W3 + b3
        
        raise NotImplementedError
        
        return z3
    
    def predict(self, X):
        """
        Make predictions using the model
        Args:
            X: Input data of shape (batch_size, n_input)
        Returns:
            Predicted class labels
        """
        logits = self.forward(X)
        probs = self.sigmoid(logits)
        return np.argmax(probs, axis=1)
    
    def get_params(self):
        """Get model parameters as a flat array"""
        # TODO: Concatenate all model parameters into a single flat array
        #
        # This function is needed for automatic differentiation (autograd).
        # The grad() function requires all parameters to be in a single array.
        params = np.concatenate([self.W1.flatten(), self.b1.flatten(), self.W2.flatten(), self.b2.flatten(), self.W3.flatten(), self.b3.flatten()])
        # print(params.shape)
        return params
        raise NotImplementedError
    
    def set_params(self, params):
        """Set model parameters from a flat array"""
        W1_size = self.n_input * self.n_hidden1
        b1_size = self.n_hidden1
        W2_size = self.n_hidden1 * self.n_hidden2
        b2_size = self.n_hidden2
        W3_size = self.n_hidden2 * self.n_output
        b3_size = self.n_output
        
        idx = 0
        self.W1 = params[idx:idx + W1_size].reshape(self.n_input, self.n_hidden1)
        idx += W1_size
        
        self.b1 = params[idx:idx + b1_size]
        idx += b1_size
        
        self.W2 = params[idx:idx + W2_size].reshape(self.n_hidden1, self.n_hidden2)
        idx += W2_size
        
        self.b2 = params[idx:idx + b2_size]
        idx += b2_size
        
        self.W3 = params[idx:idx + W3_size].reshape(self.n_hidden2, self.n_output)
        idx += W3_size
        
        self.b3 = params[idx:idx + b3_size]
