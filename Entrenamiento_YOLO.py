import os
from ultralytics import YOLO


def entrenar_mi_yolo():
    # 1. Definimos la ruta exacta al archivo data.yaml
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ruta_yaml = os.path.join(BASE_DIR, "dataset_yolo", "data.yaml")

    if not os.path.exists(ruta_yaml):
        print(f" Error: No encuentro el archivo en {ruta_yaml}")
        print("Asegúrate de que descomprimiste el ZIP en la carpeta correcta.")
        return

    print(" Archivo data.yaml encontrado. ¡Preparando la IA!")

    # 2. Cargamos el modelo base YOLOv8 Nano (el más rápido)
    model = YOLO('yolov8n.pt')

    # 3. COMENZAMOS EL ENTRENAMIENTO
    print(" Arrancando el entrenamiento. Ve a por un café, esto va a tardar un rato...")

    # Configuramos el entrenamiento para que funcione estable en Windows sin tarjeta gráfica NVIDIA
    model.train(
        data=ruta_yaml,
        epochs=20,  # 20 pasadas (suficiente para ver si aprende)
        imgsz=640,  # Resolución estándar de YOLO
        device='cpu',  # Forzamos CPU para evitar errores gráficos
        workers=1,  # Evita cuelgues de memoria en Windows
        plots=True  # ¡Te generará gráficas automáticamente para la memoria del TFG!
    )

    print(" ¡Entrenamiento completado!")
    print("Tus resultados y tu nuevo modelo están en la carpeta 'runs/detect/train'")


if __name__ == "__main__":
    entrenar_mi_yolo()