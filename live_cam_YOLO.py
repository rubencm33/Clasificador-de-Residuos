import cv2
from ultralytics import YOLO


def probar_yolo_directo():
    print("Cargando modelo YOLOv8...")
    model = YOLO('yolov8n.pt')

    # Encendemos la webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara web.")
        return

    print("🎥 CÁMARA ENCENDIDA - Muestra varios objetos (botellas, móviles, tazas...)")
    print("Pulsa 'Q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Efecto espejo

        resultados = model(frame, conf=0.5, verbose=False)

        frame_anotado = resultados[0].plot()

        # Mostramos la imagen
        cv2.imshow("YOLOv8 - Deteccion Multiple", frame_anotado)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    probar_yolo_directo()