from idlelib import history

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import os


def train_recycling_model():
    DATASET_PATH = "C:/Users/User/Downloads/DATASET"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    print("Cargando imágenes desde:", DATASET_PATH)

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_dataset.class_names
    print(f"Clases detectadas: {class_names}")

    # ==========================================
    # CREAR EL MODELO BASE CONGELADO
    # ==========================================
    print("Construyendo el modelo de Inteligencia Artificial...")
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Cerebro principal congelado

    model = models.Sequential([
        layers.Rescaling(1. / 127.5, offset=-1),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),

        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # --- FASE 1: ENTRENAMIENTO SUPERFICIAL ---
    print("\n--- FASE 1: Entrenamiento inicial (Calentamiento) ---")
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=5)

    # ==========================================
    # ✨ NUEVO: FINE-TUNING (Descongelar el cerebro)
    # ==========================================
    print("\n--- FASE 2: Fine-Tuning (Aprendizaje Profundo) ---")
    # 1. Descongelamos el cerebro principal
    base_model.trainable = True

    # 2. Volvemos a congelar las primeras 100 capas (las más básicas)
    # y dejamos que aprenda solo en las últimas 50 capas (las más complejas).
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # 3. Recompilamos con un Learning Rate MUY BAJO (0.0001) para que aprenda sin volverse loco
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 4. Entrenamos en profundidad
    print("Entrenando las capas profundas...")
    history_fine = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=10
    )

    import matplotlib.pyplot as plt

    # Unimos los datos de la Fase 1 y la Fase 2
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']

    # Creamos el lienzo de la gráfica
    plt.figure(figsize=(10, 8))

    # Gráfica 1: Precisión (Accuracy)
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Entrenamiento (Accuracy)', linewidth=2)
    plt.plot(val_acc, label='Validación (Val Accuracy)', linewidth=2)
    plt.legend(loc='lower right')
    plt.ylabel('Precisión')
    plt.title('Precisión y Pérdida durante el Entrenamiento')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Gráfica 2: Pérdida (Loss)
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Entrenamiento (Loss)', linewidth=2)
    plt.plot(val_loss, label='Validación (Val Loss)', linewidth=2)
    plt.legend(loc='upper right')
    plt.ylabel('Pérdida (Error)')
    plt.xlabel('Época')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Guardamos la imagen como archivo y la mostramos
    plt.savefig('grafica_entrenamiento.png', dpi=300, bbox_inches='tight')
    print("¡Gráfica guardada como 'grafica_entrenamiento.png'!")
    # plt.show() # (Opcional) Descomenta esto si quieres que se abra una ventana con la gráfica al terminar

    # ==========================================

    modelo_guardado = 'modelo_reciclaje.keras'
    model.save(modelo_guardado)
    print(f"\n¡Entrenamiento completado! El modelo de alta precisión se ha guardado como '{modelo_guardado}'")


if __name__ == "__main__":
    train_recycling_model()