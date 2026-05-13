import cv2
import numpy as np
import tensorflow as tf
import os
import time

# Importamos "Los Ojos" (Tu código de la Fase 1)
from src.segmentation import segment_object_grabcut
from src.morphology import clean_mask
from src.contours import get_main_object


def aplicacion_final():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # ==========================================
    # 1. CARGAMOS EL "CEREBRO" (TensorFlow)
    # ==========================================
    print("Cargando red neuronal...")
    try:
        model = tf.keras.models.load_model('modelo_reciclaje.keras')
    except:
        print("Error: No se encuentra 'modelo_reciclaje.keras'. Asegúrate de que el archivo existe en esta carpeta.")
        return
    class_names = ['CARTON', 'PLASTICO', 'VIDRIO']

    # ==========================================
    # 2. CARGAMOS LA IMAGEN
    # ==========================================
    # IMPORTANTE: Cambia esta ruta a la foto que quieras probar
    ruta_imagen = os.path.join(BASE_DIR, "data", "input", "carton.jpg")
    img = cv2.imread(ruta_imagen)

    if img is None:
        print(f"Error: No se pudo cargar la imagen en la ruta: {ruta_imagen}")
        return

    output = img.copy()

    # ==========================================
    # FASE 1: LOS OJOS (BUSCAR EL OBJETO CON OPENCV)
    # ==========================================
    print("Buscando el objeto en la imagen...")
    raw_mask = segment_object_grabcut(img)
    clean_obj_mask = clean_mask(raw_mask)
    cnt, bbox = get_main_object(clean_obj_mask)

    # Variables de diseño y variable para el recorte
    x, y, w, h = 30, 40, img.shape[1] - 60, img.shape[0] - 80
    obj_cropped = None

    if cnt is not None:
        x, y, w, h = bbox
        # Guardamos el recorte exacto del objeto detectado
        obj_cropped = img[y:y + h, x:x + w]
    else:
        print("Aviso: No se detectaron bordes claros. Analizando la foto completa.")

    # ==========================================
    # FASE 2: EL CEREBRO (CLASIFICAR CON LA IA)
    # ==========================================
    print("Analizando material...")

    # --- LA CORRECCIÓN CLAVE ---
    # Si OpenCV encontró el objeto, le pasamos a la IA SOLO el recorte (zoom al material).
    # Si no encontró nada, le pasamos la foto entera por precaución.
    if cnt is not None and obj_cropped is not None and obj_cropped.size > 0:
        imagen_a_analizar = obj_cropped
        print("Evaluando solo la textura interior del recuadro...")
    else:
        imagen_a_analizar = img

    # Convertimos y redimensionamos la imagen elegida
    img_rgb = cv2.cvtColor(imagen_a_analizar, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)

    # La IA predice
    predicciones = model.predict(img_array, verbose=0)[0]
    indice_ganador = np.argmax(predicciones)
    etiqueta = class_names[indice_ganador]
    seguridad = predicciones[indice_ganador] * 100

    # ==========================================
    # MOSTRAR Y GUARDAR EL RESULTADO FINAL
    # ==========================================
    texto_final = f"{etiqueta} ({seguridad:.1f}%)"
    print(f"¡Terminado! Objeto clasificado como: {texto_final}")

    # --- Diseño del Banner ---
    color_fondo = (0, 255, 0)  # Verde
    color_texto = (0, 0, 0)  # Negro
    grosor_caja = 3
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    escala = 0.8
    grosor_letra = 2

    if cnt is not None:
        # 1. Recuadro principal grueso
        cv2.rectangle(output, (x, y), (x + w, y + h), color_fondo, grosor_caja)

        # 2. Calcular tamaño del texto
        (ancho_texto, alto_texto), baseline = cv2.getTextSize(texto_final, fuente, escala, grosor_letra)

        # 3. Lógica Anti-Cortes: Si choca con el techo, lo dibujamos por dentro del recuadro
        espacio_necesario_arriba = alto_texto + 10

        if y - espacio_necesario_arriba >= 0:
            cv2.rectangle(output, (x, y - espacio_necesario_arriba), (x + ancho_texto, y), color_fondo, -1)
            cv2.putText(output, texto_final, (x, y - 5), fuente, escala, color_texto, grosor_letra)
        else:
            cv2.rectangle(output, (x, y), (x + ancho_texto, y + espacio_necesario_arriba), color_fondo, -1)
            cv2.putText(output, texto_final, (x, y + alto_texto + 5), fuente, escala, color_texto, grosor_letra)

    else:
        # Si falla el recorte, ponemos el texto flotando con sombra para que se lea
        cv2.putText(output, texto_final, (x + 2, y + 2), fuente, escala, (0, 0, 0), grosor_letra + 1)
        cv2.putText(output, texto_final, (x, y), fuente, escala, color_fondo, grosor_letra)

    # --- CREAR CARPETAS POR MATERIAL Y GUARDAR ALLÍ ---
    carpeta_destino = os.path.join(BASE_DIR, "outputs", etiqueta)

    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)
        print(f"📁 Nueva subcarpeta creada: {carpeta_destino}")

    timestamp = int(time.time())
    output_path = os.path.join(carpeta_destino, f"resultado_{timestamp}.png")

    cv2.imwrite(output_path, output)
    print(f"--> Imagen guardada exitosamente en: {output_path}")

    # --- Mostrar en pantalla ---
    cv2.imshow("Sistema IA - Clasificacion de Residuos", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    aplicacion_final()