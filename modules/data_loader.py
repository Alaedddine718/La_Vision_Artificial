"""
Carga y preprocesamiento del dataset CIFAR-10
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import ssl

# Fix para errores SSL en macOS
ssl._create_default_https_context = ssl._create_unverified_context


class DatasetManager:
    """Gestor del dataset CIFAR-10 con funcionalidades de preprocesamiento"""
    
    def __init__(self):
        self.num_classes = 10
        self.img_height = 32
        self.img_width = 32
        self.channels = 3
        
        # Etiquetas de las clases en español
        self.class_labels = [
            'Avión', 'Automóvil', 'Pájaro', 'Gato', 'Ciervo',
            'Perro', 'Rana', 'Caballo', 'Barco', 'Camión'
        ]
        
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        
    def load_dataset(self):
        """Carga el dataset CIFAR-10 desde Keras"""
        print("📥 Cargando dataset CIFAR-10...")
        
        (train_imgs, train_lbls), (test_imgs, test_lbls) = cifar10.load_data()
        
        self.train_images = train_imgs
        self.train_labels = train_lbls
        self.test_images = test_imgs
        self.test_labels = test_lbls
        
        print(f"✅ ¡Dataset cargado con éxito!")
        print(f"   Conjunto de entrenamiento: {self.train_images.shape}")
        print(f"   Conjunto de prueba: {self.test_images.shape}")
        
        return self
    
    def preprocess_data(self):
        """Normaliza las imágenes y convierte las etiquetas a formato one-hot"""
        print("\n🔄 Preprocesando datos...")
        
        # Normalización: píxeles de [0, 255] a [0, 1]
        self.train_images = self.train_images.astype('float32') / 255.0
        self.test_images = self.test_images.astype('float32') / 255.0
        
        # One-hot encoding de las etiquetas
        self.train_labels = to_categorical(self.train_labels, self.num_classes)
        self.test_labels = to_categorical(self.test_labels, self.num_classes)
        
        print(f"✅ ¡Preprocesamiento completado!")
        print(f"   Forma de imágenes: {self.train_images.shape}")
        print(f"   Forma de etiquetas: {self.train_labels.shape}")
        
        return self
    
    def get_training_data(self):
        """Devuelve los datos de entrenamiento"""
        return self.train_images, self.train_labels
    
    def get_test_data(self):
        """Devuelve los datos de prueba"""
        return self.test_images, self.test_labels
    
    def get_sample_images(self, num_samples=9):
        """Devuelve una muestra aleatoria de imágenes para visualización"""
        indices = np.random.choice(len(self.train_images), num_samples, replace=False)
        sample_imgs = self.train_images[indices]
        sample_lbls = np.argmax(self.train_labels[indices], axis=1)
        
        return sample_imgs, sample_lbls
    
    def get_class_name(self, class_index):
        """Devuelve el nombre de la clase dado el índice"""
        return self.class_labels[class_index]


# Función auxiliar para usar el módulo fácilmente
def prepare_cifar10_data():
    """
    Función de conveniencia que carga y preprocesa el dataset CIFAR-10
    
    Returns:
        tuple: (train_images, train_labels, test_images, test_labels, class_names)
    """
    manager = DatasetManager()
    manager.load_dataset().preprocess_data()
    
    train_x, train_y = manager.get_training_data()
    test_x, test_y = manager.get_test_data()
    
    return train_x, train_y, test_x, test_y, manager.class_labels
