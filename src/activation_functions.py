import torch

class ActivationFunction:
    #Activation function
    
    @staticmethod
    def forward(x):
        raise NotImplementedError
    
    @staticmethod
    def derivative(x):
        raise NotImplementedError


class Sigmoid(ActivationFunction):
    """Función Sigmoide: σ(x) = 1 / (1 + e^(-x))"""
    
    @staticmethod
    def forward(x):
        return 1 / (1 + torch.exp(-x))
    
    @staticmethod
    def derivative(x):
        # Derivada: σ'(x) = σ(x) * (1 - σ(x))
        s = Sigmoid.forward(x)
        return s * (1 - s)


class ReLU(ActivationFunction):
    """Función ReLU: f(x) = max(0, x)"""
    
    @staticmethod
    def forward(x):
        return torch.maximum(torch.tensor(0.0, device=x.device), x)
    
    @staticmethod
    def derivative(x):
        # Derivada: f'(x) = 1 si x > 0, else 0
        return (x > 0).float()


class Tanh(ActivationFunction):
    """Función Tangente Hiperbólica"""
    
    @staticmethod
    def forward(x):
        return torch.tanh(x)
    
    @staticmethod
    def derivative(x):
        # Derivada: f'(x) = 1 - tanh²(x)
        return 1 - torch.tanh(x) ** 2


class Softmax(ActivationFunction):
    """Función Softmax para capa de salida"""
    
    @staticmethod
    def forward(x):
        # Restar el máximo para estabilidad numérica
        exp_x = torch.exp(x - torch.max(x, dim=1, keepdim=True)[0])
        return exp_x / torch.sum(exp_x, dim=1, keepdim=True)
    
    @staticmethod
    def derivative(x):
        # Para softmax con cross-entropy, se simplifica en backprop
        return torch.ones_like(x)