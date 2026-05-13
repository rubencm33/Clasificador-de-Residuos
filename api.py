import cv2
import numpy as np
import tensorflow as tf
import os
import time
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import uvicorn

# Importamos "Los Ojos" (Tu código de la Fase 1)
from src.segmentation import segment_object_grabcut
from src.morphology import clean_mask
from src.contours import get_main_object

app = FastAPI(title="API Visual de Reciclaje", description="Sube una foto y recibe la imagen procesada con IA.")

# Cargamos el modelo
try:
    model = tf.keras.models.load_model('modelo_reciclaje.keras')
    class_names = ['CARTON', 'PLASTICO', 'VIDRIO']
    print("🧠 IA lista.")
except Exception as e:
    print(f"Error: {e}")
    model = None


@app.post("/clasificar-visual/")
async def clasificar_visual(file: UploadFile = File(...)):
    # 1. Leer imagen
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Imagen no válida"}

    output = img.copy()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # --- PROCESAMIENTO IGUAL AL ANTERIOR ---
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

    # --- DIBUJAR ETIQUETA Y RECUADRO LIMPIO (BOUNDING BOX) (CORREGIDO) ---
    texto_final = f"{etiqueta} ({seguridad:.1f}%)"

    color_verde = (0, 255, 0)  # Verde
    color_texto = (0, 0, 0)  # Negro
    grosor_linea = 2
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    escala_texto = 0.8

    if cnt is not None:
        # 1. NUEVO: Dibujamos un RECUADRO RECTO Y LIMPIO (Bounding Box)
        # en lugar de los contornos irregulares. Esto da un aspecto mucho más profesional.
        cv2.rectangle(output, (x, y), (x + w, y + h), color_verde, grosor_linea)

        # 2. Calculamos el tamaño del texto para la etiqueta
        (ancho_t, alto_t), _ = cv2.getTextSize(texto_final, fuente, escala_texto, grosor_linea)

        # 3. Colocamos la etiqueta de texto en la esquina superior izquierda del brik
        # Asegurándonos de que no se corte por el borde de la imagen
        if y - (alto_t + 10) >= 0:
            # Hay espacio arriba: Dibujar fondo y texto arriba del brik
            cv2.rectangle(output, (x, y - alto_t - 10), (x + ancho_t, y), color_verde, -1)
            cv2.putText(output, texto_final, (x, y - 5), fuente, escala_texto, color_texto, grosor_linea)
        else:
            # NO hay espacio arriba: Dibujar fondo y texto dentro del brik
            cv2.rectangle(output, (x, y), (x + ancho_t, y + alto_t + 10), color_verde, -1)
            cv2.putText(output, texto_final, (x, y + alto_t + 5), fuente, escala_texto, color_texto, grosor_linea)

    # --- GUARDADO PARA ESTADÍSTICAS (Opcional, pero recomendado) ---
    carpeta_destino = os.path.join(BASE_DIR, "outputs", etiqueta)
    if not os.path.exists(carpeta_destino): os.makedirs(carpeta_destino)
    cv2.imwrite(os.path.join(carpeta_destino, f"api_{int(time.time())}.png"), output)

    # Convertimos la imagen de OpenCV (array) a un formato que entienda el navegador (PNG)
    _, im_png = cv2.imencode(".png", output)

    # Creamos un flujo de datos (stream) para enviarlo sin guardarlo como archivo temporal
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")


@app.get("/stats/")
async def obtener_estadisticas():
    """
    Endpoint que recorre la carpeta 'outputs' y cuenta cuántos archivos
    hay en cada categoría para generar estadísticas.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path_outputs = os.path.join(BASE_DIR, "outputs")

    # Diccionario para guardar los conteos
    resumen = {
        "CARTON": 0,
        "PLASTICO": 0,
        "VIDRIO": 0,
        "TOTAL_GENERAL": 0
    }

    if not os.path.exists(path_outputs):
        return {"mensaje": "Todavía no hay datos de salida.", "stats": resumen}

    # Usamos os.walk para buscar en todas las subcarpetas (incluyendo api_uploads)
    for root, dirs, files in os.walk(path_outputs):
        for file in files:
            # Solo contamos archivos de imagen
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Buscamos si la ruta contiene alguna de nuestras palabras clave
                for categoria in ["CARTON", "PLASTICO", "VIDRIO"]:
                    if categoria in root.upper():
                        resumen[categoria] += 1
                        resumen["TOTAL_GENERAL"] += 1
                        break  # Pasamos al siguiente archivo

    return {
        "mensaje": "Estadísticas de reciclaje actualizadas",
        "datos": resumen,
        "hoy": time.strftime("%d/%m/%Y %H:%M:%S")
    }


import csv
from fastapi.responses import FileResponse


@app.get("/descargar-informe/")
async def descargar_informe():
    """
    Endpoint que genera un archivo CSV (Excel) con el recuento total
    de residuos reciclados y lo descarga en el ordenador del usuario.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path_outputs = os.path.join(BASE_DIR, "outputs")

    # 1. Hacemos el recuento de los materiales
    resumen = {"CARTON": 0, "PLASTICO": 0, "VIDRIO": 0}

    if os.path.exists(path_outputs):
        for root, dirs, files in os.walk(path_outputs):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    for categoria in resumen.keys():
                        if categoria in root.upper():
                            resumen[categoria] += 1
                            break

    # 2. Creamos el archivo CSV físico en la carpeta outputs
    ruta_csv = os.path.join(path_outputs, "informe_reciclaje.csv")

    with open(ruta_csv, mode='w', newline='', encoding='utf-8') as archivo_csv:
        writer = csv.writer(archivo_csv)
        # Escribimos las cabeceras (Los títulos de las columnas)
        writer.writerow(["Material", "Cantidad Reciclada"])

        # Escribimos los datos
        for material, cantidad in resumen.items():
            writer.writerow([material, cantidad])

        # Fila final con el total
        writer.writerow(["TOTAL_GENERAL", sum(resumen.values())])

    # 3. Devolvemos el archivo directamente para que el navegador lo descargue
    return FileResponse(ruta_csv, media_type='text/csv', filename="informe_reciclaje_IA.csv")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)