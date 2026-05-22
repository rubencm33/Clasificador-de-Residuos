# ♻️ Clasificador de Reciclaje — Smart Recycling AI

[![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow)](https://tensorflow.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=for-the-badge)](https://ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv)](https://opencv.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)

Sistema escalable de Visión Artificial e Inteligencia Artificial para la clasificación y detección automatizada de residuos domésticos (**Cartón, Plástico y Vidrio**).

El proyecto ha evolucionado a través de dos arquitecturas: una primera versión híbrida (OpenCV + MobileNetV2) y una segunda versión de detección múltiple en tiempo real basada en YOLOv8, todo ello servido a través de una API REST y un panel de control web interactivo.

---

## 🏗️ Evolución del Sistema

### Arquitectura V1 — Clasificación Híbrida (OpenCV + MobileNetV2)
- **👁️ Segmentación (Fase 1):** Aislamiento del residuo mediante el algoritmo `GrabCut` y extracción del *Bounding Box* rotado (`MinAreaRect`) con OpenCV.
- **🧠 Clasificación (Fase 2):** Red neuronal **MobileNetV2** entrenada con *Transfer Learning* y *Fine-Tuning* para detectar casos complejos como envases multicapa (Tetra Brick).
- **🎯 Precisión:** ~95% de *Validation Accuracy* sobre objetos individuales.
- **⚡ Rendimiento:** >20 FPS en CPU convencional.

### Arquitectura V2 — Detección Múltiple (YOLOv8)
- **🚀 Single-Shot Detector:** Detección simultánea de múltiples residuos en la misma escena sin necesidad de preprocesamiento con OpenCV.
- **📦 Dataset anotado:** Entrenamiento con dataset de Roboflow con miles de imágenes etiquetadas con *Bounding Boxes* y archivo `data.yaml`.
- **🌐 Dashboard Web:** Panel de control interactivo (HTML + Tailwind CSS + JavaScript) con *Drag & Drop* para clasificar imágenes desde el navegador.
- **⚡ Rendimiento:** >30 FPS en CPU, procesando escenas completas en un único pase tensorial.

---

## ✨ Características Principales

- **API REST Asíncrona:** Backend con FastAPI con endpoints de clasificación visual (streaming), datos (JSON), estadísticas y descarga de informes CSV.
- **📊 Analítica y Persistencia:** Registro automático de cada inferencia con *timestamp*. Generación de reportes de sostenibilidad en formato CSV.
- **🎥 Visión en Tiempo Real:** Módulo `live_cam.py` con zona de escaneo estática (V1) y monitorización de pantalla completa (V2).

---

## 📂 Arquitectura del Proyecto

```text
Reciclador/
├── src/                     # Algoritmos de Visión Artificial y Segmentación (V1)
├── outputs/                 # Almacenamiento histórico de analítica e informes CSV
├── api.py                   # Servidor central FastAPI (Orquestador de API y Frontend)
├── index.html               # Interfaz gráfica interactiva (Dashboard Web)
├── live_cam_YOLO.py         # Módulo de detección múltiple por webcam en tiempo real (V2)
├── Entrenamiento_YOLO.py    # Script de entrenamiento y transferencia de aprendizaje para YOLO
├── modelo_yolo_reciclaje.pt # Pesos de la red neuronal final YOLOv8 (Fase 2)
├── modelo_reciclaje.keras   # Pesos de la red neuronal MobileNetV2 (Fase 1)
├── live_cam.py              # Módulo de vídeo estático de la Fase 1
├── train_model.py           # Entrenamiento original de la Fase 1
└── requirements.txt         # Dependencias del ecosistema de software
````

🚀 Instalación y Despliegue

1. Clonar el repositorio:
```
git clone https://github.com/rubencm33/Clasificador-de-Residuos.git
cd Clasificador-de-Residuos
````
2. Crear entorno virtual e instalar dependencias:
```python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
🛠️ Modo de Uso
1. Levantar el Panel de Control Web (API + Frontend):

Ejecuta el servidor local para activar el ecosistema completo.
```
python api.py
```
Abre tu navegador y entra en: http://127.0.0.1:8000 para interactuar con la IA visualmente.

2. Detección Múltiple en Tiempo Real (YOLOv8):

Abre la cámara web para procesar múltiples residuos simultáneamente mediante la Arquitectura V2.
```
python live_cam_YOLO.py
```
3. Clasificación Clásica (MobileNetV2):

Para testear la Arquitectura V1 (Clasificación secuencial de un solo objeto).
```
python live_cam.py
```

📈 Resultados Destacados

Precisión V1: ~95% en validación aislando objetos en fondos controlados.

Precisión V2: Alta robustez en detección simultánea (mAP) e inmunidad al ruido de fondo.

Rendimiento: Latencias inferiores a 100ms, permitiendo ejecuciones superiores a 20 FPS en procesadores convencionales sin necesidad de aceleración gráfica dedicada (GPU).
