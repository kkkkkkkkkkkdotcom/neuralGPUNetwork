import torch
import numpy as np
from activation_functions import Sigmoid, ReLU, Tanh, Softmax


class NeuralNetwork:

    #Red Neuronal Multicapa,Backpropagation.PyTorch solo para operaciones matriciales en GPU.

    
    def __init__(self, layer_sizes, activation='sigmoid', learning_rate=0.01, device='cuda'):
        """
        Args:
            layer_sizes: Lista con el número de neuronas por capa [input, hidden1, hidden2, ..., output]
            activation: 'sigmoid', 'relu', 'tanh'
            learning_rate: Tasa de aprendizaje
            device: 'cuda' o 'cpu'
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.learning_rate = learning_rate
        self.device = device
        
        # Seleccionar función de activación
        self.activation_functions = {
            'sigmoid': Sigmoid,
            'relu': ReLU,
            'tanh': Tanh
        }
        self.activation = self.activation_functions[activation]
        self.output_activation = Softmax  # Para clasificación
        
        # Inicializar pesos y biases
        self.weights = []
        self.biases = []
        self._initialize_parameters()
        
        # Guardar valores intermedios para backprop
        self.z_values = []  # Pre-activación
        self.a_values = []  # Post-activación
        
    def _initialize_parameters(self):
        """Inicializar pesos con Xavier/He initialization"""
        for i in range(self.num_layers - 1):
            # Xavier initialization para sigmoid/tanh
            # He initialization para ReLU
            if isinstance(self.activation, ReLU):
                std = np.sqrt(2.0 / self.layer_sizes[i])
            else:
                std = np.sqrt(1.0 / self.layer_sizes[i])
            
            # Pesos: matriz de (layer_i x layer_i+1)
            w = torch.randn(
                self.layer_sizes[i], 
                self.layer_sizes[i + 1],
                device=self.device
            ) * std
            
            # Biases: vector de (1 x layer_i+1)
            b = torch.zeros(1, self.layer_sizes[i + 1], device=self.device)
            
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, X):

        #Forward pass manual.
        
        #Returns:
        #    Predicciones (batch_size x output_size)

        self.z_values = []
        self.a_values = [X]  # a[0] es el input
        
        # Propagar por capas ocultas
        for i in range(self.num_layers - 2):
            # z = a * W + b
            z = torch.mm(self.a_values[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            # a = activation(z)
            a = self.activation.forward(z)
            self.a_values.append(a)
        
        # Capa de salida (con softmax)
        z_out = torch.mm(self.a_values[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_out)
        
        a_out = self.output_activation.forward(z_out)
        self.a_values.append(a_out)
        
        return a_out
    
    def backward(self, X, y_true):

        #Backpropagation manual.
        #Calcula gradientes

        batch_size = X.shape[0]
        
        # Lista para guardar deltas (errores) de cada capa
        deltas = []
        
        # ===== CAPA DE SALIDA =====
        # Para softmax + cross-entropy: delta = y_pred - y_true
        y_pred = self.a_values[-1]
        delta_output = y_pred - y_true
        deltas.append(delta_output)
        
        # ===== CAPAS OCULTAS (backprop hacia atrás) =====
        for i in range(self.num_layers - 2, 0, -1):
            # delta[l] = delta[l+1] * W[l+1]^T ⊙ activation'(z[l])
            delta = torch.mm(deltas[-1], self.weights[i].t())
            delta = delta * self.activation.derivative(self.z_values[i - 1])
            deltas.append(delta)
        
        # Invertir deltas para que estén en orden correcto
        deltas.reverse()
        
        # ===== ACTUALIZAR PESOS Y BIASES =====
        for i in range(self.num_layers - 1):
            # Gradiente de pesos: dW = a[l]^T * delta[l+1]
            dW = torch.mm(self.a_values[i].t(), deltas[i]) / batch_size
            
            # Gradiente de biases: db = suma de delta[l+1]
            db = torch.sum(deltas[i], dim=0, keepdim=True) / batch_size
            
            # Actualización con descenso de gradiente
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db
    
    def train_step(self, X, y):
        #Un paso completo de entrenamiento: forward + backward
        # Forward pass
        predictions = self.forward(X)
        
        # Calcular loss (cross-entropy)
        loss = self.cross_entropy_loss(predictions, y)
        
        # Backward pass
        self.backward(X, y)
        
        return loss
    
    def cross_entropy_loss(self, y_pred, y_true):
        """Calcula Cross-Entropy Loss"""
        # Evitar log(0)
        epsilon = 1e-10
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
        
        # Loss = -sum(y_true * log(y_pred))
        loss = -torch.sum(y_true * torch.log(y_pred)) / y_true.shape[0]
        return loss
    
    def predict(self, X):
        """Predecir clases para nuevos datos"""
        predictions = self.forward(X)
        return torch.argmax(predictions, dim=1)
    
    def accuracy(self, X, y_true):
        """Calcular precisión"""
        predictions = self.predict(X)
        y_true_labels = torch.argmax(y_true, dim=1)
        correct = (predictions == y_true_labels).sum().item()
        return correct / y_true.shape[0]