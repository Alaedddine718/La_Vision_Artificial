"""
Arquitectura CNN para CIFAR-10
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, 
    MaxPooling2D, 
    Flatten, 
    Dense,
    Input
)


class ImageClassifier:
    """Clase para la construcción del modelo CNN"""
    
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        """
        Inicializa el clasificador de imágenes
        
        Args:
            input_shape: Dimensiones de la entrada (height, width, channels)
            num_classes: Número de clases a predecir
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_architecture(self):
        """Construye la arquitectura de la red neuronal"""
        print("\n🏗️  Construyendo arquitectura CNN...")
        
        network = Sequential(name="CIFAR10_Classifier")
        
        # Input layer
        network.add(Input(shape=self.input_shape, name="Input_Layer"))
        
        # ========== PRIMER BLOQUE CONVOLUCIONAL ==========
        # Extrae características de bajo nivel (bordes, esquinas, texturas)
        network.add(Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu',
            name="Convolutional_Block1"
        ))
        
        # Reduce las dimensiones espaciales manteniendo las características importantes
        network.add(MaxPooling2D(
            pool_size=(2, 2),
            name="Pooling_Block1"
        ))
        
        # ========== SEGUNDO BLOQUE CONVOLUCIONAL ==========
        # Extrae características de alto nivel (formas, patrones complejos)
        network.add(Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation='relu',
            name="Convolutional_Block2"
        ))
        
        # Reducción dimensional adicional
        network.add(MaxPooling2D(
            pool_size=(2, 2),
            name="Pooling_Block2"
        ))
        
        # ========== CLASIFICADOR FULLY-CONNECTED ==========
        # Aplana los feature maps en un vector 1D
        network.add(Flatten(name="Feature_Flattening"))
        
        # Capa densa para combinar las características extraídas
        network.add(Dense(
            units=64,
            activation='relu',
            name="Dense_Layer"
        ))
        
        # Capa de salida con softmax para probabilidades de las clases
        network.add(Dense(
            units=self.num_classes,
            activation='softmax',
            name="Classification_Output"
        ))
        
        self.model = network
        
        print("✅ ¡Arquitectura construida con éxito!")
        return self
    
    def compile_model(self, optimizer='adam', learning_rate=None):
        """
        Compila el modelo con función de pérdida y optimizador
        
        Args:
            optimizer: Nombre del optimizador a utilizar
            learning_rate: Learning rate personalizado (opcional)
        """
        if self.model is None:
            raise ValueError("¡El modelo debe ser construido antes de ser compilado!")
        
        print(f"\n⚙️  Compilando modelo con optimizador: {optimizer}")
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ ¡Modelo compilado!")
        return self
    
    def display_summary(self):
        """Muestra el resumen de la arquitectura"""
        if self.model is None:
            raise ValueError("¡El modelo debe ser construido antes de visualizar el resumen!")
        
        print("\n" + "="*70)
        print("ARQUITECTURA DE LA RED NEURONAL")
        print("="*70)
        self.model.summary()
        print("="*70 + "\n")
        
        return self
    
    def get_model(self):
        """Devuelve el modelo Keras"""
        return self.model
    
    def count_parameters(self):
        """Cuenta los parámetros totales del modelo"""
        if self.model is None:
            raise ValueError("¡El modelo debe ser construido!")
        
        total_params = self.model.count_params()
        return total_params


# Función auxiliar para crear rápidamente el modelo
def create_cifar10_classifier():
    """
    Crea, compila y devuelve un modelo CNN listo para el entrenamiento
    
    Returns:
        keras.Model: Modelo compilado
    """
    classifier = ImageClassifier()
    classifier.build_architecture()
    classifier.compile_model()
    classifier.display_summary()
    
    return classifier.get_model()
