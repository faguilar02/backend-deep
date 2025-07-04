import os
import tensorflow as tf

# Obtener ruta absoluta
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "clasificacion_model_v4.h5")
OUTPUT_PATH = os.path.join(MODEL_DIR, "clasificacion_model_v4_opt.tflite")

model = tf.keras.models.load_model(MODEL_PATH)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)
print(f"✅ Modelo optimizado guardado en: {OUTPUT_PATH}")