# ♻️ Smart Recycling AI

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![Ultralytics YOLO](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6F00?style=for-the-badge&logo=yolo)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=for-the-badge&logo=fastapi)

Sistema escalable de Inteligencia Artificial diseñado para la clasificación y detección automatizada de residuos domésticos (**Cartón, Plástico y Vidrio**). 

Este proyecto documenta una evolución arquitectónica de ingeniería: comenzando con un **Pipeline Híbrido (Fase 1)** basado en segmentación matemática y MobileNetV2, y evolucionando hacia una **Detección Unificada de un solo disparo (Fase 2)** mediante el entrenamiento de un modelo YOLOv8 personalizado, todo orquestado a través de una API REST interactiva.

## ✨ Características y Evolución Técnica

* **🧠 Arquitectura V1 (Clasificación Aislada):** Uso de `GrabCut` (OpenCV) para la extracción de la ROI y un modelo *MobileNetV2* optimizado mediante *Fine-Tuning* para inferencia ultraligera.
* **🚀 Arquitectura V2 (Detección Múltiple):** Modelo YOLOv8 entrenado a medida para detectar múltiples envases simultáneamente en la misma escena, superando las limitaciones espaciales y de fugas de segmentación de la V1.
* **⚡ API REST Asíncrona:** Backend de alto rendimiento (FastAPI) que integra *Middlewares* CORS y endpoints estadísticos.
* **🖥️ Frontend Dashboard Web:** Interfaz gráfica *Glassmorphism* servida directamente por la API que permite la inferencia visual interactiva (Drag & Drop) y la monitorización en tiempo real del impacto ecológico.
* **📊 Analítica Integrada:** Generación automatizada de reportes de sostenibilidad en formato CSV.

## 📂 Arquitectura del Proyecto

```text
Reciclador/
├── outputs/                 # Informes CSV e imágenes procesadas
├── src/                     # Módulos core de Visión Artificial (V1)
├── dataset_yolo/            # Datos etiquetados para entrenamiento (Ignorado en Git)
├── api.py                   # Servidor principal FastAPI y Frontend Web
├── index.html               # Interfaz gráfica interactiva
├── Entrenamiento_YOLO.py    # Script de entrenamiento arquitectónico V2
├── live_cam_YOLO.py         # Inferencia multi-objeto en tiempo real
├── modelo_yolo_reciclaje.pt # Pesos de la red YOLOv8
├── modelo_reciclaje.keras   # Pesos de la red MobileNetV2
└── requirements.txt
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
