import tensorflow as tf
import numpy as np
import cv2
import os


def probar_ia():
    # 1. Cargar el "cerebro" que acabas de entrenar
    print("Despertando a la IA...")
    try:
        model = tf.keras.models.load_model('../modelo_reciclaje.keras')
    except:
        print("Error: No se encuentra 'modelo_reciclaje.keras'. Asegúrate de estar en la carpeta correcta.")
        return

    # IMPORTANTE: TensorFlow ordenó tus carpetas alfabéticamente en el entrenamiento
    class_names = ['CARTON', 'PLASTICO', 'VIDRIO']

    # 2. Cargar una imagen de prueba (¡Usa una foto que la IA no haya visto antes!)
    # Escribe aquí la ruta a una foto de tu móvil o descargada de internet
    ruta_imagen = os.path.join(BASE_DIR, "../data", "input", "vino.jpg")

    if not os.path.exists(ruta_imagen):
        print(f"No encuentro la imagen de prueba en: {ruta_imagen}")
        return

    img = cv2.imread(ruta_imagen)

    # OpenCV lee los colores en BGR, pero nuestra IA estudió en RGB. ¡Hay que girarlos!
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. Preparar la imagen (La IA solo entiende fotos cuadradas de 224x224)
    img_resized = cv2.resize(img_rgb, (224, 224))

    # Convertimos la foto en un formato que TensorFlow entienda (añadimos una dimensión de "lote")
    img_array = np.expand_dims(img_resized, axis=0)

    # ==========================================
    # 4. LA HORA DE LA VERDAD (INFERENCIA)
    # ==========================================
    print("Analizando imagen...")
    predicciones = model.predict(img_array)

    # predicciones es una lista de 3 porcentajes. Ej: [0.01, 0.98, 0.01] (98% seguro de que es Plástico)
    indice_ganador = np.argmax(predicciones)  # Busca qué número de los 3 es el más alto
    etiqueta_ganadora = class_names[indice_ganador]
    seguridad = np.max(predicciones) * 100

    print("\n" + "=" * 40)
    print(f"🤖 LA IA DICE QUE ES: {etiqueta_ganadora}")
    print(f"📊 Nivel de seguridad:  {seguridad:.2f} %")
    print("=" * 40 + "\n")

    # Mostrar la foto en pantalla
    cv2.imshow(f"Prediccion: {etiqueta_ganadora}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    probar_ia()