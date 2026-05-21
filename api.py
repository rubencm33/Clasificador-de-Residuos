import cv2
import numpy as np
import tensorflow as tf
import os
import time
import io
import csv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Importamos "Los Ojos" (Tu código de la Fase 1)
from src.segmentation import segment_object_grabcut
from src.morphology import clean_mask
from src.contours import get_main_object

app = FastAPI(title="API Visual de Reciclaje", description="Sube una foto y recibe la imagen procesada con IA.")

# --- CONFIGURACIÓN DE SEGURIDAD (CORS) ---
# Esto permite que la interfaz web se comunique con la API sin que el navegador la bloquee
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cargamos el modelo
try:
    model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'modelo_reciclaje.keras'))
    class_names = ['CARTON', 'PLASTICO', 'VIDRIO']
    print("🧠 IA lista.")
except Exception as e:
    print(f"Error: {e}")
    model = None


# ==========================================
# ENDPOINT 1: LA INTERFAZ WEB (FRONTEND)
# ==========================================
@app.get("/")
async def mostrar_interfaz():
    """Sirve la página web principal al usuario."""
    ruta_html = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(ruta_html):
        return FileResponse(ruta_html)
    return {"error": "No se encuentra el archivo index.html en la carpeta principal."}


# ==========================================
# ENDPOINT 2: CLASIFICACIÓN CON IA
# ==========================================
@app.post("/clasificar-visual/")
async def clasificar_visual(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Imagen no válida"}

    output = img.copy()

    # --- PROCESAMIENTO ---
    raw_mask = segment_object_grabcut(img)
    clean_obj_mask = clean_mask(raw_mask)
    cnt, bbox = get_main_object(clean_obj_mask)

    x, y, w, h = 30, 40, img.shape[1] - 60, img.shape[0] - 80
    obj_cropped = None

    if cnt is not None:
        x, y, w, h = bbox
        margen = 40
        y_inicio = max(0, y - margen)
        y_fin = min(img.shape[0], y + h + margen)
        x_inicio = max(0, x - margen)
        x_fin = min(img.shape[1], x + w + margen)
        obj_cropped = img[y_inicio:y_fin, x_inicio:x_fin]
    else:
        obj_cropped = img

    if cnt is not None and obj_cropped is not None and obj_cropped.size > 0:
        imagen_a_analizar = obj_cropped
    else:
        imagen_a_analizar = img

    img_rgb = cv2.cvtColor(imagen_a_analizar, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)

    predicciones = model.predict(img_array, verbose=0)[0]
    indice_ganador = np.argmax(predicciones)
    etiqueta = class_names[indice_ganador]
    seguridad = float(predicciones[indice_ganador] * 100)

    # --- DIBUJAR ---
    texto_final = f"{etiqueta} ({seguridad:.1f}%)"
    color_verde = (0, 255, 0)
    color_texto = (0, 0, 0)
    grosor_linea = 2
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    escala_texto = 0.8

    if cnt is not None:
        cv2.rectangle(output, (x, y), (x + w, y + h), color_verde, grosor_linea)
        (ancho_t, alto_t), _ = cv2.getTextSize(texto_final, fuente, escala_texto, grosor_linea)

        if y - (alto_t + 10) >= 0:
            cv2.rectangle(output, (x, y - alto_t - 10), (x + ancho_t, y), color_verde, -1)
            cv2.putText(output, texto_final, (x, y - 5), fuente, escala_texto, color_texto, grosor_linea)
        else:
            cv2.rectangle(output, (x, y), (x + ancho_t, y + alto_t + 10), color_verde, -1)
            cv2.putText(output, texto_final, (x, y + alto_t + 5), fuente, escala_texto, color_texto, grosor_linea)

    # --- GUARDAR ESTADÍSTICAS ---
    carpeta_destino = os.path.join(BASE_DIR, "outputs", etiqueta)
    os.makedirs(carpeta_destino, exist_ok=True)
    cv2.imwrite(os.path.join(carpeta_destino, f"api_{int(time.time())}.png"), output)

    _, im_png = cv2.imencode(".png", output)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")


# ==========================================
# ENDPOINT 3: ESTADÍSTICAS PARA EL DASHBOARD
# ==========================================
@app.get("/stats/")
async def obtener_estadisticas():
    path_outputs = os.path.join(BASE_DIR, "outputs")
    resumen = {"CARTON": 0, "PLASTICO": 0, "VIDRIO": 0, "TOTAL_GENERAL": 0}

    if not os.path.exists(path_outputs):
        return {"mensaje": "Sin datos", "datos": resumen}

    for root, dirs, files in os.walk(path_outputs):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                for categoria in ["CARTON", "PLASTICO", "VIDRIO"]:
                    if categoria in root.upper():
                        resumen[categoria] += 1
                        resumen["TOTAL_GENERAL"] += 1
                        break
    return {"datos": resumen}


# ==========================================
# ENDPOINT 4: DESCARGAR CSV
# ==========================================
@app.get("/descargar-informe/")
async def descargar_informe():
    path_outputs = os.path.join(BASE_DIR, "outputs")
    resumen = {"CARTON": 0, "PLASTICO": 0, "VIDRIO": 0}

    if os.path.exists(path_outputs):
        for root, dirs, files in os.walk(path_outputs):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    for categoria in resumen.keys():
                        if categoria in root.upper():
                            resumen[categoria] += 1
                            break

    ruta_csv = os.path.join(path_outputs, "informe_reciclaje.csv")
    os.makedirs(path_outputs, exist_ok=True)

    with open(ruta_csv, mode='w', newline='', encoding='utf-8') as archivo_csv:
        writer = csv.writer(archivo_csv)
        writer.writerow(["Material", "Cantidad Reciclada"])
        for material, cantidad in resumen.items():
            writer.writerow([material, cantidad])
        writer.writerow(["TOTAL_GENERAL", sum(resumen.values())])

    return FileResponse(ruta_csv, media_type='text/csv', filename="informe_reciclaje_IA.csv")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)