import cv2
import numpy as np
import tensorflow as tf
import os


def aplicacion_camara():
    # 1. CARGAMOS EL "CEREBRO"
    print("Cargando red neuronal... (Esto puede tardar unos segundos)")
    try:
        model = tf.keras.models.load_model('modelo_reciclaje.keras')
    except:
        print("Error: No se encuentra 'modelo_reciclaje.keras'.")
        return
    class_names = ['CARTON', 'PLASTICO', 'VIDRIO']

    # 2. ENCENDEMOS LA CÁMARA (El número 0 suele ser la webcam principal del portátil)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara web.")
        return

    print("\n" + "=" * 40)
    print("🎥 CÁMARA ENCENDIDA")
    print("👉 Pon el objeto dentro del recuadro azul.")
    print("👉 Pulsa la tecla 'Q' para salir.")
    print("=" * 40 + "\n")

    # Bucle infinito: Leer imagen por imagen a toda velocidad
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Hacemos un "espejo" de la cámara para que sea más natural mover las cosas
        frame = cv2.flip(frame, 1)
        alto, ancho = frame.shape[:2]

        # ==========================================
        # DEFINIR LA ZONA DE ESCANEO (Un cuadrado en el centro)
        # ==========================================
        tamaño_caja = 280
        x1 = int(ancho / 2 - tamaño_caja / 2)
        y1 = int(alto / 2 - tamaño_caja / 2)
        x2 = x1 + tamaño_caja
        y2 = y1 + tamaño_caja

        # Dibujamos el recuadro "Guía" en color Cian
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, "PON EL OBJETO AQUI", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Recortamos exactamente lo que hay dentro de la zona de escaneo
        zona_escaneo = frame[y1:y2, x1:x2]

        # ==========================================
        # EL CEREBRO: ANALIZAR LA ZONA EN TIEMPO REAL
        # ==========================================
        # Preparamos el recorte para la IA
        img_rgb = cv2.cvtColor(zona_escaneo, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_array = np.expand_dims(img_resized, axis=0)

        # La IA predice el fotograma actual
        predicciones = model.predict(img_array, verbose=0)[0]
        indice_ganador = np.argmax(predicciones)
        seguridad = predicciones[indice_ganador] * 100
        etiqueta = class_names[indice_ganador]

        # ==========================================
        # MOSTRAR RESULTADO (Solo si está seguro)
        # ==========================================
        # Para que no parpadee constantemente analizando el fondo vacío,
        # le decimos que solo ponga el texto si está más de un 80% seguro.
        if seguridad > 80.0:
            texto_final = f"{etiqueta} ({seguridad:.1f}%)"

            # Dibujamos un fondo verde para el texto
            cv2.rectangle(frame, (x1, y2), (x2, y2 + 40), (0, 255, 0), -1)
            cv2.putText(frame, texto_final, (x1 + 10, y2 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            # Cambiamos el borde del recuadro a verde para dar "Feedback" de que lo ha cazado
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

        # Mostramos la ventana final
        cv2.imshow("Clasificador de Reciclaje en Directo", frame)

        # Detectar si el usuario pulsa la tecla 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Cerrando cámara...")
            break

    # Apagar la cámara y cerrar ventanas limpiamente
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    aplicacion_camara()