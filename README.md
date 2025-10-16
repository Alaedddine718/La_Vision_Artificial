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

## 🚀 Instalación

```bash
git clone https://github.com/Alaedddine718/La_Vision_Artificial.git
cd La_Vision_Artificial
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 💻 Uso

### Interfaz Web
```bash
streamlit run app.py
```

### Script de Terminal
```bash
python main.py
```

---

## 📊 Funcionalidades

- Visualización de ejemplos del dataset
- Entrenamiento del modelo
- Clasificación de imágenes personalizadas
- Gráficos de precisión y pérdida

---

## 🔧 Tecnologías

- TensorFlow/Keras
- Streamlit
- NumPy
- Matplotlib

---

## 📄 Estructura

```
La_Vision_Artificial/
├── modules/
│   ├── data_loader.py
│   ├── network.py
│   └── trainer.py
├── app.py
├── main.py
└── requirements.txt
```

---

## 📈 Resultados

- Precisión: ~70% en conjunto de prueba
- Dataset: 60.000 imágenes CIFAR-10
- Entrenamiento: 5-10 épocas

---

## 📄 Licencia

MIT License - Ver archivo LICENSE
