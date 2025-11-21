import torch
import cv2
import numpy as np
from pathlib import Path


class FaceDataLoader:
    """Cargador del dataset AT&T Faces"""
    
    def __init__(self, dataset_path, img_size=(92, 112), device='cuda'):
        """
        Args:
            dataset_path: Ruta al dataset (ej: 'datasets/att_faces')
            img_size: Tamaño de las imágenes (ancho, alto)
            device: 'cuda' o 'cpu'
        """
        self.dataset_path = Path(dataset_path)
        self.img_size = img_size
        self.device = device
        self.num_classes = 40  # AT&T tiene 40 personas
        
    def load_data(self, train_images_per_person=7):
        """
        Cargar y dividir datos en train/test.
        
        Args:
            train_images_per_person: Cuántas imágenes usar para train (de 10 total)
        
        Returns:
            X_train, y_train, X_test, y_test (todos como tensores de PyTorch)
        """
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        
        # Iterar por cada persona (s1, s2, ..., s40)
        for person_id in range(1, self.num_classes + 1):
            person_folder = self.dataset_path / f's{person_id}'
            
            # Cargar las 10 imágenes de esta persona
            images = []
            for img_num in range(1, 11):
                img_path = person_folder / f'{img_num}.pgm'
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                
                # Redimensionar si es necesario
                if img.shape != self.img_size[::-1]:
                    img = cv2.resize(img, self.img_size)
                
                # Normalizar a [0, 1]
                img = img.astype(np.float32) / 255.0
                
                # Aplanar la imagen
                img_flat = img.flatten()
                images.append(img_flat)
            
            # Dividir en train/test
            X_train.extend(images[:train_images_per_person])
            y_train.extend([person_id - 1] * train_images_per_person)  # Labels de 0 a 39
            
            X_test.extend(images[train_images_per_person:])
            y_test.extend([person_id - 1] * (10 - train_images_per_person))
        
        # Convertir a tensores de PyTorch
        X_train = torch.tensor(np.array(X_train), dtype=torch.float32, device=self.device)
        X_test = torch.tensor(np.array(X_test), dtype=torch.float32, device=self.device)
        
        # One-hot encoding de labels
        y_train = self._one_hot_encode(y_train)
        y_test = self._one_hot_encode(y_test)
        
        print(f"Dataset cargado:")
        print(f"  Train: {X_train.shape[0]} imágenes")
        print(f"  Test: {X_test.shape[0]} imágenes")
        print(f"  Input size: {X_train.shape[1]} píxeles")
        
        return X_train, y_train, X_test, y_test
    
    def _one_hot_encode(self, labels):
        """Convertir labels a one-hot encoding"""
        one_hot = torch.zeros(len(labels), self.num_classes, device=self.device)
        for i, label in enumerate(labels):
            one_hot[i, label] = 1
        return one_hot