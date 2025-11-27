# scripts/experimentos.py
import torch
import sys
sys.path.append('.')
from src import NeuralNetwork, FaceDataLoader
from pathlib import Path
import json

def experimento(config, nombre):
    """
    Ejecutar un experimento con configuración específica
    """
    print("\n" + "="*70)
    print(f"🧪 EXPERIMENTO: {nombre}")
    print("="*70)
    print(f"Configuración:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-"*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Cargar datos
    loader = FaceDataLoader('datasets/att_faces', device=device)
    X_train, y_train, X_test, y_test = loader.load_data(
        train_images_per_person=config.get('train_images', 7)
    )
    
    # Crear red neuronal
    input_size = X_train.shape[1]
    nn = NeuralNetwork(
        layer_sizes=config['architecture'],
        activation=config['activation'],
        learning_rate=config['learning_rate'],
        device=device
    )
    
    # Entrenar
    epochs = config['epochs']
    batch_size = config['batch_size']
    
    best_test_acc = 0
    
    for epoch in range(epochs):
        # Shuffle
        indices = torch.randperm(X_train.shape[0], device=device)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        # Mini-batches
        num_batches = X_train.shape[0] // batch_size
        epoch_loss = 0
        
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]
            loss = nn.train_step(X_batch, y_batch)
            epoch_loss += loss.item()
        
        # Evaluar
        train_acc = nn.accuracy(X_train, y_train)
        test_acc = nn.accuracy(X_test, y_test)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        # Mostrar progreso
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} - "
                  f"Loss: {epoch_loss/num_batches:.4f} - "
                  f"Train: {train_acc:.4f} - Test: {test_acc:.4f}")
    
    print("-"*70)
    print(f"✅ Mejor Test Accuracy: {best_test_acc:.4f} ({best_test_acc*100:.2f}%)")
    print("="*70)
    
    return best_test_acc

def ejecutar_experimentos():
    """
    Probar diferentes configuraciones
    """
    resultados = {}
    
    # ===== EXPERIMENTO 1: Baseline (tu configuración actual) =====
    config1 = {
        'architecture': [10304, 128, 64, 40],
        'activation': 'sigmoid',
        'learning_rate': 0.1,
        'epochs': 100,
        'batch_size': 32,
        'train_images': 7
    }
    resultados['1_baseline'] = experimento(config1, "Baseline (Actual)")
    
    # ===== EXPERIMENTO 2: ReLU en vez de Sigmoid =====
    config2 = config1.copy()
    config2['activation'] = 'relu'
    resultados['2_relu'] = experimento(config2, "Con ReLU")
    
    # ===== EXPERIMENTO 3: Learning Rate más bajo =====
    config3 = config1.copy()
    config3['learning_rate'] = 0.05
    resultados['3_lr_bajo'] = experimento(config3, "Learning Rate 0.05")
    
    # ===== EXPERIMENTO 4: Arquitectura más profunda =====
    config4 = config1.copy()
    config4['architecture'] = [10304, 256, 128, 64, 40]
    resultados['4_profunda'] = experimento(config4, "Arquitectura Profunda")
    
    # ===== EXPERIMENTO 5: Más epochs =====
    config5 = config1.copy()
    config5['epochs'] = 200
    resultados['5_mas_epochs'] = experimento(config5, "200 Epochs")
    
    # ===== EXPERIMENTO 6: Batch size más pequeño =====
    config6 = config1.copy()
    config6['batch_size'] = 16
    resultados['6_batch_16'] = experimento(config6, "Batch Size 16")
    
    # ===== EXPERIMENTO 7: Combinación óptima =====
    config7 = {
        'architecture': [10304, 256, 128, 64, 40],
        'activation': 'relu',
        'learning_rate': 0.05,
        'epochs': 150,
        'batch_size': 16,
        'train_images': 7
    }
    resultados['7_combinado'] = experimento(config7, "Combinación Óptima")
    
    # ===== RESUMEN =====
    print("\n\n" + "="*70)
    print("📊 RESUMEN DE EXPERIMENTOS")
    print("="*70)
    
    resultados_ordenados = sorted(resultados.items(), key=lambda x: x[1], reverse=True)
    
    for i, (nombre, acc) in enumerate(resultados_ordenados, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{emoji} {nombre:20s}: {acc:.4f} ({acc*100:.2f}%)")
    
    print("="*70)
    
    # Guardar resultados
    Path('results').mkdir(exist_ok=True)
    with open('results/experimentos.json', 'w') as f:
        json.dump(resultados, f, indent=2)
    
    print("\n💾 Resultados guardados en: results/experimentos.json")

if __name__ == '__main__':
    ejecutar_experimentos()