# 🖼️ Clasificador CNN para CIFAR-10

Red Neuronal Convolucional para la clasificación de imágenes del dataset CIFAR-10.

**Autores:** Alessio Cicilano & Alaeddine Daoudi  
**Fecha:** Octubre 2025  

🔗 [Repositorio en GitHub](https://github.com/Alaedddine718/La_Vision_Artificial)


---

## 📋 Descripción

Implementación de una CNN para clasificar imágenes del dataset CIFAR-10 en 10 categorías: Avión, Automóvil, Pájaro, Gato, Ciervo, Perro, Rana, Caballo, Barco y Camión.

---

## 🏗️ Arquitectura del Modelo

**Bloques Convolucionales:**
- Conv2D (32 filtros) + ReLU + MaxPooling
- Conv2D (64 filtros) + ReLU + MaxPooling

**Clasificador:**
- Flatten
- Dense (64 neuronas) + ReLU
- Dense (10 neuronas) + Softmax

**Total de parámetros**: ~167.500

---

## 🚀 Instalación y Ejecución

### 📱 **En macOS / Linux**

Abre el Terminal y ejecuta:

```bash
# 1. Clonar el repositorio
git clone https://github.com/Alaedddine718/La_Vision_Artificial.git
cd La_Vision_Artificial

# 2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
streamlit run app.py
```

### 🪟 **En Windows**

Abre Command Prompt (CMD) o PowerShell y ejecuta:

```bash
# 1. Clonar el repositorio
git clone https://github.com/Alaedddine718/La_Vision_Artificial.git
cd La_Vision_Artificial

# 2. Crear entorno virtual
python -m venv venv
venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
streamlit run app.py
```

---

## ⏱️ Tiempo de Instalación

- **Clonación del repositorio**: ~10 segundos
- **Creación del entorno virtual**: ~30 segundos
- **Instalación de dependencias**: ~5-10 minutos (TensorFlow es pesado)
- **Total**: ~10-15 minutos

---

## 💻 Uso

### 🌐 Opción 1: Aplicación Web (Streamlit)

Después de ejecutar `streamlit run app.py`, la aplicación se abrirá automáticamente en tu navegador en:

**http://localhost:8501**

**Funcionalidades:**

**1. 📊 Visualizar Dataset**
- Muestra 9 imágenes aleatorias del dataset CIFAR-10
- Click en "MOSTRAR EJEMPLOS"

**2. 🚀 Entrenar Modelo**
- Entrena el modelo CNN con un solo click
- Visualiza gráficos de precisión y pérdida en tiempo real
- Guarda el modelo entrenado automáticamente

**3. 🔍 Clasificar Imagen**
- Sube tu propia imagen (JPG, PNG, JPEG)
- El modelo predice la clase
- Muestra la confianza y probabilidades de todas las clases

### 📓 Opción 2: Google Colab (Sin instalación)

**¡La forma más rápida de probar el proyecto!**

1. Abre el archivo `CIFAR10_CNN_Colab.ipynb`
2. Súbelo a [Google Colab](https://colab.research.google.com/)
3. Ejecuta las celdas en orden
4. ¡Disfruta del entrenamiento con GPU gratis!

**Ventajas de Google Colab:**
- ✅ No requiere instalación local
- ✅ GPU gratuita para entrenamiento rápido
- ✅ Todo en un solo notebook interactivo
- ✅ Perfecto para aprendizaje y experimentación

---

## 📦 Requisitos

- **Python**: 3.9 o superior
- **Git**: Para clonar el repositorio
- **Conexión a Internet**: Para descargar dependencias
- **Espacio en disco**: ~2GB (incluyendo dependencias)

---

## 🔧 Tecnologías

- TensorFlow/Keras
- Streamlit
- NumPy
- Matplotlib
- Pillow

---

## 📁 Estructura del Proyecto

```
La_Vision_Artificial/
├── modules/
│   ├── __init__.py
│   ├── data_loader.py      # Carga y preprocesamiento
│   ├── network.py           # Arquitectura CNN
│   └── trainer.py           # Entrenamiento
├── checkpoints/             # Modelos guardados (generado)
├── app.py                   # Interfaz web Streamlit
├── main.py                  # Script de terminal
├── CIFAR10_CNN_Colab.ipynb  # Notebook para Google Colab
├── requirements.txt         # Dependencias
├── LICENSE                  # Licencia MIT
└── README.md               # Este archivo
```

---

## 📈 Resultados Esperados

- **Precisión**: ~70% en conjunto de prueba
- **Dataset**: 60.000 imágenes CIFAR-10 (50k train, 10k test)
- **Tiempo de entrenamiento**: ~5-10 minutos (5 épocas)
- **Formato de imágenes**: 32x32 píxeles RGB
- **Optimizador**: Adam
- **Función de pérdida**: Categorical Crossentropy

---

## 🎯 Características Principales

✅ **Interfaz web interactiva** con Streamlit  
✅ **Visualización en tiempo real** del entrenamiento  
✅ **Clasificación de imágenes propias**  
✅ **Modelo pre-entrenado** incluido  
✅ **Gráficos de rendimiento** automáticos  
✅ **Código modular y organizado**  

---

## 🛠️ Solución de Problemas

### Error: "Python no encontrado"
```bash
# Mac/Linux
brew install python3

# Windows
Descarga Python desde: https://www.python.org/downloads/
```

### Error: "Git no encontrado"
```bash
# Mac
brew install git

# Windows
Descarga Git desde: https://git-scm.com/download/win
```

### Error de permisos en Mac
```bash
sudo chown -R $(whoami) venv
```

---

## 🚫 Detener la Aplicación

Presiona `Ctrl + C` en el terminal donde está ejecutándose Streamlit.

---

## 📄 Licencia

MIT License - Ver archivo LICENSE

---

## 👥 Autores

- **Alessio Cicilano**
- **Alaeddine Daoudi**

---

## 🌐 Repositorio

https://github.com/Alaedddine718/La_Vision_Artificial
