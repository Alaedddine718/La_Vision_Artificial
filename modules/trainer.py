"""
Entrenamiento y evaluación del modelo
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class ModelTrainer:
    """Clase para gestionar el entrenamiento y evaluación del modelo"""
    
    def __init__(self, model):
        """
        Inicializa el entrenador
        
        Args:
            model: Modelo Keras a entrenar
        """
        self.model = model
        self.training_history = None
        self.evaluation_results = None
        
    def train(self, train_x, train_y, epochs=10, batch_size=64, validation_split=0.1, verbose=1):
        """
        Entrena el modelo con los datos proporcionados
        
        Args:
            train_x: Imágenes de entrenamiento
            train_y: Etiquetas de entrenamiento
            epochs: Número de épocas
            batch_size: Tamaño del batch
            validation_split: Porcentaje de datos para validación
            verbose: Nivel de detalle
            
        Returns:
            History: Objeto history de Keras con métricas de entrenamiento
        """
        print("\n" + "="*70)
        print("🚀 INICIO DEL ENTRENAMIENTO")
        print("="*70)
        print(f"Épocas: {epochs}")
        print(f"Tamaño de batch: {batch_size}")
        print(f"División de validación: {validation_split*100}%")
        print(f"Muestras de entrenamiento: {len(train_x)}")
        print("="*70 + "\n")
        
        start_time = datetime.now()
        
        self.training_history = self.model.fit(
            train_x, 
            train_y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "="*70)
        print(f"✅ ENTRENAMIENTO COMPLETADO en {duration:.2f} segundos")
        print("="*70 + "\n")
        
        return self.training_history
    
    def evaluate(self, test_x, test_y, verbose=1):
        """
        Evalúa el modelo en el conjunto de prueba
        
        Args:
            test_x: Imágenes de prueba
            test_y: Etiquetas de prueba
            verbose: Nivel de detalle
            
        Returns:
            dict: Diccionario con pérdida y precisión
        """
        print("\n" + "="*70)
        print("🧪 EVALUACIÓN EN EL CONJUNTO DE PRUEBA")
        print("="*70)
        
        test_loss, test_accuracy = self.model.evaluate(test_x, test_y, verbose=verbose)
        
        self.evaluation_results = {
            'loss': test_loss,
            'accuracy': test_accuracy
        }
        
        print("\n" + "="*70)
        print("📊 RESULTADOS FINALES:")
        print(f"   Pérdida: {test_loss:.4f}")
        print(f"   Precisión: {test_accuracy*100:.2f}%")
        print("="*70 + "\n")
        
        return self.evaluation_results
    
    def plot_training_history(self, save_path=None):
        """
        Visualiza los gráficos de precisión y pérdida durante el entrenamiento
        
        Args:
            save_path: Ruta donde guardar el gráfico (opcional)
        """
        if self.training_history is None:
            raise ValueError("¡El modelo debe ser entrenado antes de visualizar los gráficos!")
        
        history = self.training_history.history
        
        # Extrae las métricas
        train_acc = history['accuracy']
        val_acc = history['val_accuracy']
        train_loss = history['loss']
        val_loss = history['val_loss']
        epochs_range = range(1, len(train_acc) + 1)
        
        # Crea la figura con 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Gráfico Precisión
        ax1.plot(epochs_range, train_acc, 'b-o', label='Entrenamiento', linewidth=2, markersize=6)
        ax1.plot(epochs_range, val_acc, 'r-s', label='Validación', linewidth=2, markersize=6)
        ax1.set_title('Precisión durante el Entrenamiento', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Época', fontsize=12)
        ax1.set_ylabel('Precisión', fontsize=12)
        ax1.legend(loc='lower right', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Gráfico Pérdida
        ax2.plot(epochs_range, train_loss, 'b-o', label='Entrenamiento', linewidth=2, markersize=6)
        ax2.plot(epochs_range, val_loss, 'r-s', label='Validación', linewidth=2, markersize=6)
        ax2.set_title('Pérdida durante el Entrenamiento', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Época', fontsize=12)
        ax2.set_ylabel('Pérdida', fontsize=12)
        ax2.legend(loc='upper right', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📁 Gráfico guardado en: {save_path}")
        
        plt.show()
        
        return fig
    
    def get_final_metrics(self):
        """Devuelve las métricas finales de entrenamiento y validación"""
        if self.training_history is None:
            raise ValueError("¡El modelo debe ser entrenado!")
        
        history = self.training_history.history
        
        metrics = {
            'final_train_accuracy': history['accuracy'][-1],
            'final_val_accuracy': history['val_accuracy'][-1],
            'final_train_loss': history['loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'best_val_accuracy': max(history['val_accuracy']),
            'best_epoch': np.argmax(history['val_accuracy']) + 1
        }
        
        return metrics
    
    def save_model(self, filepath):
        """
        Guarda el modelo entrenado
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        self.model.save(filepath)
        print(f"💾 Modelo guardado en: {filepath}")


# Función auxiliar para entrenamiento completo
def train_and_evaluate_model(model, train_x, train_y, test_x, test_y, 
                            epochs=10, batch_size=64, show_plots=True):
    """
    Ejecuta entrenamiento y evaluación completos del modelo
    
    Args:
        model: Modelo a entrenar
        train_x, train_y: Datos de entrenamiento
        test_x, test_y: Datos de prueba
        epochs: Número de épocas
        batch_size: Tamaño del batch
        show_plots: Si True, muestra los gráficos
        
    Returns:
        tuple: (trainer, history, results)
    """
    trainer = ModelTrainer(model)
    
    # Entrenamiento
    history = trainer.train(train_x, train_y, epochs=epochs, batch_size=batch_size)
    
    # Evaluación
    results = trainer.evaluate(test_x, test_y)
    
    # Visualiza gráficos
    if show_plots:
        trainer.plot_training_history()
    
    # Muestra métricas finales
    metrics = trainer.get_final_metrics()
    print("\n📈 MÉTRICAS FINALES:")
    print(f"   Precisión Entrenamiento: {metrics['final_train_accuracy']*100:.2f}%")
    print(f"   Precisión Validación: {metrics['final_val_accuracy']*100:.2f}%")
    print(f"   Mejor Precisión Validación: {metrics['best_val_accuracy']*100:.2f}% (época {metrics['best_epoch']})")
    
    return trainer, history, results
