# ♻️ Clasificador de Reciclaje

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=for-the-badge&logo=fastapi)

Sistema híbrido de Visión Artificial e Inteligencia Artificial diseñado para la clasificación automatizada de residuos domésticos (**Cartón, Plástico y Vidrio**). 

Este proyecto implementa una arquitectura en dos fases: una primera etapa de preprocesamiento y segmentación (aislamiento del objeto) y una segunda etapa de clasificación profunda utilizando Transfer Learning, todo ello servido a través de una API REST de alto rendimiento.

## ✨ Características Principales

* **👁️ Visión Artificial (Fase 1):** Eliminación del ruido de fondo y segmentación del residuo utilizando el algoritmo `GrabCut` y extracción del *Bounding Box* (`MinAreaRect`) mediante OpenCV.
* **🧠 Deep Learning (Fase 2):** Clasificación del material mediante una red neuronal **MobileNetV2** optimizada mediante *Fine-Tuning* para detectar casos complejos (como envases multicapa o briks).
* **⚡ API REST Asíncrona:** Backend desarrollado con FastAPI para inferencia remota, con endpoints para clasificación visual (streaming) y de datos (JSON).
* **🎥 Visión en Tiempo Real:** Módulo `live_cam.py` optimizado para funcionar a más de 20 FPS en procesadores convencionales.
* **📊 Analítica y Persistencia:** Generación automatizada de reportes de sostenibilidad en formato CSV.

## 📂 Arquitectura del Proyecto

```text
Reciclador/
├── outputs/                 # Informes CSV e imágenes procesadas
├── src/                     # Módulos core de Visión Artificial
│   ├── contours.py
│   ├── morphology.py
│   └── segmentation.py
├── tools/                   # Scripts auxiliares (Data Augmentation)
│   └── extraer_fotos.py
├── api.py                   # Servidor principal FastAPI
├── Clasificar_foto.py       # Script de prueba local de imágenes
├── live_cam.py              # Inferencia en directo por webcam
├── train_model.py           # Script de reentrenamiento de la red
├── modelo_reciclaje.keras   # Modelo de IA compilado
├── grafica_entrenamiento.png
└── requirements.txt
```
🚀 Instalación y Despliegue

Clonar el repositorio:
```
git clone [https://github.com/rubencm33/Clasificador-de-Residuos.git
cd Clasificador-de-Residuos
```
Crear entorno virtual e instalar dependencias:
```
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
🛠️ Modo de Uso
El sistema cuenta con tres vías principales de ejecución dependiendo de la necesidad:

1. Desplegar la API Web:
Levanta el servidor local para procesar peticiones web.
```
python api.py
(Documentación interactiva disponible en: http://127.0.0.1:8000/docs)
```
2. Cámara en Tiempo Real:
Abre la webcam para detectar materiales en directo. Sitúa el objeto dentro de la zona delimitada.
```
python live_cam.py
```
3. Clasificación de Foto Local:
Analiza una foto guardada en el disco para comprobar la precisión y el Bounding Box.
```
python Clasificar_foto.py
```

## 🛠️ Modo de Uso Detallado

El sistema cuenta con tres vías principales de ejecución dependiendo de la necesidad del usuario:

### 1. Desplegar la API Web (FastAPI)
Levanta el servidor local para procesar peticiones web. Es ideal para integrar el modelo con aplicaciones móviles o páginas web.
```bash
python api.py
```
¿Cómo interactuar con la API?

Abre tu navegador y ve a: http://127.0.0.1:8000/docs

Verás la interfaz gráfica (Swagger UI) generada automáticamente.
```
Despliega el endpoint POST /clasificar-visual/, pulsa en "Try it out", sube una foto de un residuo y pulsa "Execute".
```
La API te devolverá la misma foto con el Bounding Box y el porcentaje de seguridad dibujados.

Puedes usar los endpoints GET /stats/ o GET /descargar-informe/ para ver o descargar el CSV con las estadísticas de reciclaje.

2. Cámara en Tiempo Real (Live Cam)
Inicia la webcam para detectar materiales en directo. Utiliza un umbral estricto (>80% de seguridad) para evitar falsos positivos y parpadeos en pantalla.

```
python live_cam.py
```
Instrucciones de uso:

Sitúa el objeto dentro del recuadro central delimitado en la pantalla.

Cuando la IA esté segura, el recuadro se volverá verde y mostrará la etiqueta del material (Cartón, Plástico o Vidrio).

Pulsa la tecla Q en tu teclado para cerrar la cámara de forma segura.

3. Clasificación de Foto Local (Test Rápido)
Script diseñado para analizar una única fotografía guardada en el disco duro. Perfecto para comprobar el rendimiento del Bounding Box y la precisión de la red neuronal paso a paso.
```
python Clasificar_foto.py
(Nota: Para probar diferentes imágenes, simplemente modifica la variable ruta_imagen dentro del código del script apuntando a la foto deseada).
```
