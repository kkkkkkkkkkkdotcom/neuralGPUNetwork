# src/train.py
import torch
from .neural_network import NeuralNetwork
from .data_loader import FaceDataLoader
import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_history(history):
    """Visualizar progreso del entrenamiento"""
    # Crear carpeta si no existe
    Path('results/plots').mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history['train_loss'])
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['test_acc'], label='Test')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/plots/training_history.png', dpi=150)
    print("\n📊 Gráficas guardadas en results/plots/")


def train_network():
    # Verificar GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")
    
    # Cargar datos
    loader = FaceDataLoader('datasets/att_faces', device=device)
    X_train, y_train, X_test, y_test = loader.load_data(train_images_per_person=7)
    
    # Crear red neuronal: 10304 -> 128 -> 64 -> 40
    input_size = X_train.shape[1]  # 92 * 112 = 10304 píxeles
    nn = NeuralNetwork(
        layer_sizes=[input_size, 128, 64, 40],
        activation='sigmoid',
        learning_rate=0.1,
        device=device
    )
    
    # Entrenar
    epochs = 100
    batch_size = 32
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
    print("\n🚀 Iniciando entrenamiento...\n")
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = torch.randperm(X_train.shape[0], device=device)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        # Mini-batch training
        epoch_loss = 0
        num_batches = X_train.shape[0] // batch_size
        
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            
            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]
            
            loss = nn.train_step(X_batch, y_batch)
            epoch_loss += loss.item()
        
        # Calcular métricas
        avg_loss = epoch_loss / num_batches
        train_acc = nn.accuracy(X_train, y_train)
        test_acc = nn.accuracy(X_test, y_test)
        
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        # Imprimir progreso cada 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - "
                  f"Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f}")
    
    print("\n✅ Entrenamiento completado!")
    print(f"Precisión final en test: {history['test_acc'][-1]:.4f}")
    
    # Guardar modelo
    torch.save({
        'weights': nn.weights,
        'biases': nn.biases,
        'layer_sizes': nn.layer_sizes
    }, 'results/model.pth')
    
    # Graficar resultados
    plot_training_history(history)
    
    return nn, history


if __name__ == '__main__':
    model, history = train_network()