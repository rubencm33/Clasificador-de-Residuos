import cv2
import os


def video_a_fotos():
    # 1. Graba un vídeo de 10-15 segundos enfocando el brik desde varios ángulos
    # 2. Pasa el vídeo al ordenador y pon la ruta aquí:
    ruta_video = "C:/Users/User/Downloads/botella2.mp4"

    # 3. La carpeta donde quieres guardar las fotos nuevas:
    carpeta_destino = "C:/Users/User/Downloads/DATASET/VIDRIO"

    cap = cv2.VideoCapture(ruta_video)

    if not cap.isOpened():
        print("Error: No se pudo abrir el vídeo. Revisa la ruta.")
        return

    contador_frames = 0
    fotos_guardadas = 0

    print("Extrayendo fotos del vídeo...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # El vídeo ha terminado

        # Guardamos 1 de cada 5 frames (para que las fotos no sean 100% idénticas)
        if contador_frames % 5 == 0:
            nombre_foto = os.path.join(carpeta_destino, f"brik_extra_{fotos_guardadas}.jpg")
            cv2.imwrite(nombre_foto, frame)
            fotos_guardadas += 1

        contador_frames += 1

    cap.release()
    print(f"¡Magia! Se han extraído y guardado {fotos_guardadas} fotos nuevas en {carpeta_destino}")


if __name__ == "__main__":
    video_a_fotos()