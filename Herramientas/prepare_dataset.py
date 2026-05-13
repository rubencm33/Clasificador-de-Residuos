import os
import shutil


def prepare_dataset():
    RUTA_ORIGEN = "C:/Users/User/Downloads/DATASET"
    RUTA_DESTINO = "dataset"

    print("Iniciando la preparación del dataset...")

    for nueva_categoria, carpetas_originales in mapeo.items():
        ruta_nueva_categoria = os.path.join(RUTA_DESTINO, nueva_categoria)
        os.makedirs(ruta_nueva_categoria, exist_ok=True)

        contador = 0
        for carpeta_orig in carpetas_originales:
            ruta_origen_completa = os.path.join(RUTA_ORIGEN, carpeta_orig)

            if not os.path.exists(ruta_origen_completa):
                print(f"  ⚠️ No se encontró la carpeta original: {carpeta_orig}")
                continue

            for archivo in os.listdir(ruta_origen_completa):
                if archivo.lower().endswith(('.jpg', '.jpeg', '.png')):
                    origen = os.path.join(ruta_origen_completa, archivo)
                    nuevo_nombre = f"{nueva_categoria}_{contador}.jpg"
                    destino = os.path.join(ruta_nueva_categoria, nuevo_nombre)

                    shutil.copy(origen, destino)
                    contador += 1

        print(f"✅ {nueva_categoria}: {contador} imágenes procesadas.")

    print(f"\n¡Proceso terminado! Tu dataset está listo en la carpeta '{RUTA_DESTINO}'.")


if __name__ == "__main__":
    prepare_dataset()