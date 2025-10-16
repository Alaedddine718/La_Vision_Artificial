"""
Entrenamiento del modelo CNN en CIFAR-10
Autores: Alessio Cicilano & Alaeddine Daoudi
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce los warnings de TensorFlow

from modules.data_loader import prepare_cifar10_data
from modules.network import create_cifar10_classifier
from modules.trainer import train_and_evaluate_model


def main():
    """Función principal para ejecutar el entrenamiento completo"""
    
    print("\n" + "="*70)
    print("🎯 CLASIFICADOR DE IMÁGENES CIFAR-10")
    print("    Red Neuronal Convolucional (CNN)")
    print("="*70)
    
    # ========== FASE 1: CARGA DE DATOS ==========
    print("\n[FASE 1/4] Carga y preprocesamiento del dataset...")
    train_x, train_y, test_x, test_y, class_names = prepare_cifar10_data()
    
    print(f"\n📊 Información sobre el dataset:")
    print(f"   Clases: {', '.join(class_names)}")
    print(f"   Número de clases: {len(class_names)}")
    
    # ========== FASE 2: CONSTRUCCIÓN DEL MODELO ==========
    print("\n[FASE 2/4] Construcción del modelo CNN...")
    model = create_cifar10_classifier()
    
    # ========== FASE 3: ENTRENAMIENTO ==========
    print("\n[FASE 3/4] Entrenamiento del modelo...")
    trainer, history, results = train_and_evaluate_model(
        model=model,
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        epochs=10,
        batch_size=64,
        show_plots=True
    )
    
    # ========== FASE 4: GUARDADO ==========
    print("\n[FASE 4/4] Guardando el modelo...")
    
    # Crea directorio si no existe
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    model_path = 'checkpoints/cifar10_classifier.h5'
    trainer.save_model(model_path)
    
    # ========== RESUMEN FINAL ==========
    print("\n" + "="*70)
    print("✅ ¡ENTRENAMIENTO COMPLETADO CON ÉXITO!")
    print("="*70)
    print(f"\n📌 Precisión final en el conjunto de prueba: {results['accuracy']*100:.2f}%")
    print(f"📌 Pérdida final en el conjunto de prueba: {results['loss']:.4f}")
    print(f"📌 Modelo guardado en: {model_path}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
