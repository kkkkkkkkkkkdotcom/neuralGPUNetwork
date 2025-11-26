# Neural Network with Backpropagation

A custom implementation of a multi-layer neural network with manual backpropagation for facial recognition using the AT&T Faces dataset. This project implements neural network fundamentals from scratch using PyTorch primarily for GPU-accelerated matrix operations.

## 🎯 Project Overview

This project demonstrates a **from-scratch neural network implementation** designed for facial recognition tasks. The implementation includes:

- **Manual backpropagation algorithm** (not using PyTorch's autograd)
- **Multiple activation functions** (Sigmoid, ReLU, Tanh, Softmax)
- **GPU acceleration** via PyTorch tensors
- **AT&T Faces dataset** support (40 subjects, 10 images each)

## 📁 Project Structure

```
src/
├── __init__.py              # Package initialization
├── neural_network.py        # Core neural network implementation
├── activation_functions.py  # Activation function classes
├── data_loader.py          # AT&T Faces dataset loader
└── train.py                # Training script and visualization
```

## 🧠 Core Components

### 1. Neural Network (`neural_network.py`)

The `NeuralNetwork` class implements a fully-connected feedforward neural network with manual backpropagation.

**Key Features:**
- Configurable architecture (arbitrary number of layers and neurons)
- Xavier/He weight initialization
- Manual forward and backward propagation
- Cross-entropy loss calculation
- Prediction and accuracy evaluation

**Architecture:**
```python
NeuralNetwork(
    layer_sizes=[10304, 128, 64, 40],  # Input → Hidden1 → Hidden2 → Output
    activation='sigmoid',               # Activation function
    learning_rate=0.1,                 # Learning rate
    device='cuda'                      # GPU/CPU device
)
```

**Main Methods:**
- `forward(X)` - Forward propagation through the network
- `backward(X, y_true)` - Backpropagation and weight updates
- `train_step(X, y)` - Complete training step (forward + backward)
- `predict(X)` - Predict class labels
- `accuracy(X, y_true)` - Calculate classification accuracy

### 2. Activation Functions (`activation_functions.py`)

Implements multiple activation functions with their derivatives:

| Function | Formula | Use Case |
|----------|---------|----------|
| **Sigmoid** | σ(x) = 1 / (1 + e^(-x)) | Hidden layers (smooth gradients) |
| **ReLU** | f(x) = max(0, x) | Hidden layers (faster training) |
| **Tanh** | f(x) = tanh(x) | Hidden layers (zero-centered) |
| **Softmax** | σ(x)ᵢ = e^(xᵢ) / Σe^(xⱼ) | Output layer (classification) |

Each class provides:
- `forward(x)` - Apply activation function
- `derivative(x)` - Compute derivative for backpropagation

### 3. Data Loader (`data_loader.py`)

The `FaceDataLoader` class handles the AT&T Faces dataset:

**Features:**
- Loads grayscale face images (92×112 pixels)
- Normalizes pixel values to [0, 1]
- Flattens images to 1D vectors (10,304 features)
- Splits data into train/test sets
- One-hot encodes labels (40 classes)
- GPU tensor conversion

**Usage:**
```python
loader = FaceDataLoader('datasets/att_faces', device='cuda')
X_train, y_train, X_test, y_test = loader.load_data(train_images_per_person=7)
```

**Dataset Split:**
- Training: 7 images per person (280 total)
- Testing: 3 images per person (120 total)

### 4. Training Script (`train.py`)

Complete training pipeline with visualization:

**Training Configuration:**
- Epochs: 100
- Batch size: 32
- Mini-batch gradient descent
- Data shuffling per epoch

**Features:**
- Real-time training metrics
- Loss and accuracy tracking
- Model checkpointing
- Training history visualization

**Outputs:**
- Trained model saved to `results/model.pth`
- Training plots saved to `results/plots/training_history.png`

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch opencv-python numpy matplotlib
```

### Dataset Setup

1. Download the AT&T Faces dataset
2. Extract to `datasets/att_faces/`
3. Structure should be:
   ```
   datasets/att_faces/
   ├── s1/
   │   ├── 1.pgm
   │   ├── 2.pgm
   │   └── ...
   ├── s2/
   └── ...
   ```

### Training the Model

```python
from src.train import train_network

# Train the network
model, history = train_network()

# Check final accuracy
print(f"Test Accuracy: {history['test_acc'][-1]:.4f}")
```

### Using the Trained Model

```python
from src import NeuralNetwork, FaceDataLoader

# Load data
loader = FaceDataLoader('datasets/att_faces')
X_train, y_train, X_test, y_test = loader.load_data()

# Create and train model
nn = NeuralNetwork(
    layer_sizes=[10304, 128, 64, 40],
    activation='sigmoid',
    learning_rate=0.1
)

# Make predictions
predictions = nn.predict(X_test)
accuracy = nn.accuracy(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")
```

## 🔬 Technical Details

### Backpropagation Algorithm

The implementation uses the standard backpropagation algorithm:

1. **Forward Pass:**
   - z^(l) = a^(l-1) · W^(l) + b^(l)
   - a^(l) = activation(z^(l))

2. **Backward Pass:**
   - δ^(output) = y_pred - y_true (for softmax + cross-entropy)
   - δ^(l) = δ^(l+1) · W^(l+1)^T ⊙ activation'(z^(l))

3. **Weight Updates:**
   - ∇W^(l) = a^(l-1)^T · δ^(l) / batch_size
   - W^(l) ← W^(l) - learning_rate · ∇W^(l)

### Weight Initialization

- **Xavier Initialization** (Sigmoid/Tanh): std = √(1/n_in)
- **He Initialization** (ReLU): std = √(2/n_in)

### Loss Function

Cross-entropy loss for multi-class classification:

```
L = -Σ(y_true · log(y_pred)) / batch_size
```

## 📊 Performance

Expected results on AT&T Faces dataset:
- Training accuracy: ~95-100%
- Test accuracy: ~85-95%
- Training time: ~2-5 minutes on GPU

## 🎓 Educational Purpose

This project is designed for learning neural network fundamentals:

- ✅ Understanding forward/backward propagation
- ✅ Implementing gradient descent from scratch
- ✅ Working with activation functions
- ✅ Handling real-world image data
- ✅ GPU acceleration with PyTorch tensors

## 📝 License

This is an educational project for AI coursework - Facial Recognition.

## 🤝 Contributing

This is a learning project. Feel free to fork and experiment with:
- Different network architectures
- Additional activation functions
- Regularization techniques (dropout, L2)
- Optimization algorithms (momentum, Adam)
- Data augmentation strategies
