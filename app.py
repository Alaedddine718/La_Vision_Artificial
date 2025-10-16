"""
Interfaz Web CIFAR-10
Autores: Alessio Cicilano & Alaeddine Daoudi
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from modules.data_loader import DatasetManager
from modules.network import ImageClassifier
from modules.trainer import ModelTrainer
import os
from PIL import Image
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuración de página
st.set_page_config(
    page_title="Clasificador CNN CIFAR-10",
    page_icon="🖼️",
    layout="wide"
)

# Estilo personalizado
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


def show_header():
    """Muestra el encabezado de la aplicación"""
    st.markdown('<h1 class="main-header">🖼️ Clasificador de Imágenes CIFAR-10</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🎯 **10 Clases** de objetos")
    with col2:
        st.info("🧠 **CNN** Deep Learning")
    with col3:
        st.info("📊 **60K** imágenes")


def show_dataset_info():
    """Muestra información sobre el dataset"""
    with st.expander("📚 Información sobre el Dataset CIFAR-10", expanded=False):
        st.write("""
        **CIFAR-10** es un dataset de referencia para visión por computadora, que contiene:
        
        - **60.000 imágenes** a color (32x32 píxeles)
        - **10 categorías**: Avión, Automóvil, Pájaro, Gato, Ciervo, Perro, Rana, Caballo, Barco, Camión
        - **50.000 imágenes** de entrenamiento
        - **10.000 imágenes** de prueba
        
        Cada imagen es de tamaño pequeño pero suficiente para entrenar modelos CNN eficientes.
        """)


def show_architecture_info():
    """Muestra información sobre la arquitectura"""
    with st.expander("🏗️ Arquitectura de la Red Neuronal", expanded=False):
        st.write("""
        **Arquitectura CNN de 2 bloques:**
        
        1. **Bloque Convolucional 1**
           - Conv2D: 32 filtros 3x3 + ReLU
           - MaxPooling: 2x2
           
        2. **Bloque Convolucional 2**
           - Conv2D: 64 filtros 3x3 + ReLU
           - MaxPooling: 2x2
           
        3. **Clasificador**
           - Flatten
           - Dense: 64 neuronas + ReLU
           - Dense: 10 neuronas + Softmax
           
        **Total de parámetros**: ~167.562
        """)
        
        # Diagrama de arquitectura
        st.image("https://miro.medium.com/max/1400/1*vkQ0hXDaQv57sALXAJquxA.jpeg", 
                caption="Ejemplo de arquitectura CNN", use_column_width=True)


def visualize_sample_images(manager):
    """Visualiza imágenes de ejemplo del dataset"""
    st.subheader("🖼️ Ejemplos del Dataset")
    
    sample_imgs, sample_lbls = manager.get_sample_images(9)
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for idx, ax in enumerate(axes.flat):
        # Desnormaliza si es necesario
        img = sample_imgs[idx]
        if img.max() <= 1.0:
            img = img
        
        ax.imshow(img)
        ax.set_title(manager.get_class_name(sample_lbls[idx]), 
                    fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)


def predict_image_interface():
    """Interfaz para clasificar imágenes subidas por el usuario"""
    st.header("🔍 Clasificar tu Imagen")
    
    st.write("Sube una imagen y el modelo entrenado predecirá a qué clase pertenece.")
    
    # Verificar si existe un modelo entrenado
    model_path = 'checkpoints/cifar10_classifier.h5'
    
    if not os.path.exists(model_path):
        st.warning("⚠️ No hay un modelo entrenado. Por favor, entrena el modelo primero en la pestaña 'Entrenar Modelo'.")
        return
    
    # Subir imagen
    uploaded_file = st.file_uploader(
        "Selecciona una imagen (JPG, PNG, JPEG)",
        type=['jpg', 'png', 'jpeg']
    )
    
    if uploaded_file is not None:
        # Mostrar imagen subida
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Imagen Original")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("🎯 Predicción")
            
            with st.spinner("Analizando imagen..."):
                # Cargar modelo
                model = tf.keras.models.load_model(model_path)
                
                # Preprocesar imagen
                # Redimensionar a 32x32 como CIFAR-10
                img_resized = image.resize((32, 32))
                img_array = np.array(img_resized)
                
                # Convertir a RGB si es necesario
                if img_array.shape[-1] == 4:  # RGBA
                    img_array = img_array[:, :, :3]
                elif len(img_array.shape) == 2:  # Escala de grises
                    img_array = np.stack([img_array] * 3, axis=-1)
                
                # Normalizar
                img_array = img_array.astype('float32') / 255.0
                
                # Añadir dimensión batch
                img_batch = np.expand_dims(img_array, axis=0)
                
                # Predecir
                predictions = model.predict(img_batch, verbose=0)
                predicted_class = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class] * 100
                
                # Nombres de las clases
                class_names = [
                    'Avión', 'Automóvil', 'Pájaro', 'Gato', 'Ciervo',
                    'Perro', 'Rana', 'Caballo', 'Barco', 'Camión'
                ]
                
                # Mostrar resultado
                st.success(f"**Clase Predicha:** {class_names[predicted_class]}")
                st.metric(label="Confianza", value=f"{confidence:.2f}%")
                
                # Mostrar imagen redimensionada
                st.write("**Imagen procesada (32x32px):**")
                st.image(img_resized, width=150)
        
        # Mostrar todas las probabilidades
        st.subheader("📊 Probabilidades de todas las clases")
        
        class_names = [
            'Avión', 'Automóvil', 'Pájaro', 'Gato', 'Ciervo',
            'Perro', 'Rana', 'Caballo', 'Barco', 'Camión'
        ]
        
        # Crear gráfico de barras
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#1f77b4' if i != predicted_class else '#2ca02c' for i in range(10)]
        bars = ax.barh(class_names, predictions[0] * 100, color=colors)
        ax.set_xlabel('Probabilidad (%)', fontsize=12)
        ax.set_title('Distribución de Probabilidades', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Añadir valores en las barras
        for i, (bar, prob) in enumerate(zip(bars, predictions[0] * 100)):
            if prob > 5:  # Solo mostrar si es > 5%
                ax.text(prob + 1, i, f'{prob:.1f}%', va='center', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)


def train_model_interface():
    """Interfaz para el entrenamiento del modelo"""
    st.header("🚀 Entrenamiento del Modelo")
    
    st.write("Haz clic en el botón para entrenar el modelo en el dataset CIFAR-10.")
    
    # Botón para iniciar el entrenamiento
    if st.button("🎯 INICIAR ENTRENAMIENTO", type="primary", use_container_width=True):
        
        # Barra de progreso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Fase 1: Carga de datos
            status_text.text("📥 Cargando dataset...")
            progress_bar.progress(20)
            
            manager = DatasetManager()
            manager.load_dataset().preprocess_data()
            train_x, train_y = manager.get_training_data()
            test_x, test_y = manager.get_test_data()
            
            # Fase 2: Construcción del modelo
            status_text.text("🏗️ Construyendo modelo...")
            progress_bar.progress(40)
            
            classifier = ImageClassifier()
            classifier.build_architecture().compile_model()
            model = classifier.get_model()
            
            # Fase 3: Entrenamiento
            status_text.text("🚀 Entrenamiento en curso...")
            progress_bar.progress(60)
            
            trainer = ModelTrainer(model)
            
            history = trainer.train(
                train_x, train_y,
                epochs=5,
                batch_size=64,
                validation_split=0.1,
                verbose=0
            )
            
            progress_bar.progress(80)
            
            # Fase 4: Evaluación
            status_text.text("🧪 Evaluando modelo...")
            results = trainer.evaluate(test_x, test_y, verbose=0)
            
            progress_bar.progress(100)
            status_text.text("✅ ¡Entrenamiento completado!")
            
            # Muestra resultados
            st.success("🎉 ¡Entrenamiento completado con éxito!")
            
            # Métricas finales
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="📊 Precisión Test",
                    value=f"{results['accuracy']*100:.2f}%"
                )
            
            with col2:
                st.metric(
                    label="📉 Pérdida Test",
                    value=f"{results['loss']:.4f}"
                )
            
            with col3:
                final_metrics = trainer.get_final_metrics()
                st.metric(
                    label="🏆 Mejor Precisión Val",
                    value=f"{final_metrics['best_val_accuracy']*100:.2f}%"
                )
            
            # Gráficos
            st.subheader("📈 Evolución del Entrenamiento")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Precisión
            ax1.plot(history.history['accuracy'], 'b-o', label='Entrenamiento', linewidth=2)
            ax1.plot(history.history['val_accuracy'], 'r-s', label='Validación', linewidth=2)
            ax1.set_title('Precisión', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Época')
            ax1.set_ylabel('Precisión')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Pérdida
            ax2.plot(history.history['loss'], 'b-o', label='Entrenamiento', linewidth=2)
            ax2.plot(history.history['val_loss'], 'r-s', label='Validación', linewidth=2)
            ax2.set_title('Pérdida', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Época')
            ax2.set_ylabel('Pérdida')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Opción para guardar el modelo
            if st.button("💾 Guardar Modelo"):
                if not os.path.exists('checkpoints'):
                    os.makedirs('checkpoints')
                model_path = 'checkpoints/cifar10_web_model.h5'
                trainer.save_model(model_path)
                st.success(f"Modelo guardado en: {model_path}")
            
        except Exception as e:
            st.error(f"❌ Error durante el entrenamiento: {str(e)}")
            st.exception(e)


def main():
    """Función principal de la aplicación"""
    
    # Encabezado
    show_header()
    
    # Información
    show_dataset_info()
    show_architecture_info()
    
    # Pestañas para diferentes funcionalidades
    tab1, tab2, tab3 = st.tabs(["📊 Visualizar Dataset", "🚀 Entrenar Modelo", "🔍 Clasificar Imagen"])
    
    with tab1:
        st.subheader("Explora el Dataset CIFAR-10")
        st.write("Visualiza algunas imágenes de ejemplo del dataset para ver los datos con los que se entrenará el modelo.")
        
        if st.button("🖼️ MOSTRAR EJEMPLOS", type="primary", use_container_width=True):
            with st.spinner("Cargando..."):
                manager = DatasetManager()
                manager.load_dataset().preprocess_data()
                visualize_sample_images(manager)
    
    with tab2:
        train_model_interface()
    
    with tab3:
        predict_image_interface()
    
    # Pie de página
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Desarrollado por Alessio Cicilano & Alaeddine Daoudi | 2025</p>
            <p>🧠 Deep Learning | 🖼️ Visión por Computadora | 🐍 Python + TensorFlow</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
