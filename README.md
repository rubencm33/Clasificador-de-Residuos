# ♻️ Smart Recycling AI — Ecosistema de Visión Artificial e IA para Reciclaje

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![Ultralytics YOLO](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6F00?style=for-the-badge&logo=yolo)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=for-the-badge&logo=fastapi)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv)

Este repositorio contiene el desarrollo completo de un sistema inteligente y escalable para la clasificación y detección múltiple de residuos domésticos (**Cartón, Plástico y Vidrio**). 

El proyecto documenta una evolución de ingeniería en dos fases fundamentales: comenzando con un **Pipeline Híbrido secuencial (Fase 1)** y escalando hacia una arquitectura de **Detección Unificada de múltiples objetos en tiempo real (Fase 2)** integrada con un panel de control web interactivo.

---

## 🌟 FASE 2 (Actual): Detección Múltiple y Dashboard Web

La arquitectura definitiva del sistema elimina las dependencias geométricas tradicionales y unifica el proceso de inferencia para entornos reales y complejos.

* **🚀 Inferencia Multi-Objeto (YOLOv8):** Mediante el entrenamiento de un modelo personalizado (`modelo_yolo_reciclaje.pt`, generado desde `best.pt`), el sistema localiza, encuadra y clasifica múltiples residuos simultáneamente en la misma escena, superando las limitaciones de oclusión y ruido de fondo.
* **🖥️ Panel de Control Web Interactivo (Frontend):** Interfaz gráfica moderna con diseño *Glassmorphism* servida directamente por la API. Permite a los usuarios arrastrar y soltar imágenes (*Drag & Drop*) para obtener la inferencia visual con *Bounding Boxes* y porcentajes de seguridad renderizados al vuelo.
* **📊 Persistencia y Analítica:** Servidor asíncrono (**FastAPI**) que monitoriza el impacto ecológico en tiempo real, actualiza los contadores del Dashboard y genera informes administrativos descargables en formato CSV.

---

## 🧠 FASE 1 (MVP): Clasificación Híbrida Aislada

La primera iteración del proyecto se diseñó como un Producto Mínimo Viable enfocado en la eficiencia local:
* **Segmentación Clásica:** Implementación del algoritmo `GrabCut` (OpenCV) y operaciones morfológicas para aislar el residuo del entorno y extraer la Región de Interés (ROI).
* **Clasificación Convolucional:** Red neuronal ligera *MobileNetV2* optimizada mediante *Transfer Learning* y *Fine-Tuning* para clasificar objetos individuales en hardware de consumo común (CPU).

---

## 📂 Arquitectura del Repositorio

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
