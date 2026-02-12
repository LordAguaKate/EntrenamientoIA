import cv2
import numpy as np
import tensorflow.lite as tflite
import time
from pathlib import Path

# --- CONFIGURACIÓN ---
# Obtener la ruta absoluta basada en la ubicación de ESTE archivo script
BASE_DIR = Path(__file__).resolve().parent.parent # Sube de 'tests/' a 'entrenamientoia/'
MODEL_PATH = BASE_DIR / "models" / "modelo_residuos_rpi.tflite"
CONFIDENCE_THRESHOLD = 0.50  # Solo mostrar predicción si la confianza es > 50%

# Las 30 clases en el orden exacto del entrenamiento
CLASSES = [
    'aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans', 'cardboard_boxes', 
    'cardboard_packaging', 'clothing', 'coffee_grounds', 'disposable_plastic_cutlery', 
    'eggshells', 'food_waste', 'glass_beverage_bottles', 'glass_cosmetic_containers', 
    'glass_food_jars', 'magazines', 'newspaper', 'office_paper', 'paper_cups', 
    'plastic_cup_lids', 'plastic_detergent_bottles', 'plastic_food_containers', 
    'plastic_shopping_bags', 'plastic_soda_bottles', 'plastic_straws', 'plastic_trash_bags', 
    'plastic_water_bottles', 'shoes', 'steel_food_cans', 'styrofoam_cups', 
    'styrofoam_food_containers', 'tea_bags'
]

class WasteClassifier:
    def __init__(self, model_path):
        print(f"Cargando modelo desde: {model_path}")
        try:
            self.interpreter = tflite.Interpreter(model_path=str(model_path))
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_shape = self.input_details[0]['shape'][1:3] # (224, 224)
            print("Modelo cargado exitosamente.")
        except Exception as e:
            print(f"Error crítico cargando el modelo: {e}")
            exit(1)

    def predict(self, frame):
        # 1. Preprocesamiento
        # Redimensionar a 224x224 (lo que espera MobileNetV2)
        resized_frame = cv2.resize(frame, (self.input_shape[0], self.input_shape[1]))
        
        # Convertir BGR (OpenCV) a RGB (Entrenamiento)
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        # Normalizar (0 a 1) y agregar dimensión de batch
        input_data = np.expand_dims(rgb_frame, axis=0).astype(np.float32)
        input_data = input_data / 255.0

        # 2. Inferencia
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        # 3. Decodificar resultados
        idx = np.argmax(output_data)
        confidence = output_data[idx]
        
        return CLASSES[idx], confidence

def main():
    classifier = WasteClassifier(MODEL_PATH)
    
    # Iniciar cámara (0 suele ser la webcam por defecto)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    print("--- INICIANDO SISTEMA DE DETECCIÓN ---")
    print("Presiona 'q' para salir.")

    # Variables para calcular FPS
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error capturando frame.")
            break

        # Realizar predicción
        label, confidence = classifier.predict(frame)

        # Cálculo de FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # --- DIBUJAR INTERFAZ (UI) ---
        
        # Color del texto basado en la confianza (Verde si es seguro, Rojo si duda)
        color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 0, 255)
        display_text = f"{label}: {confidence*100:.1f}%"
        
        if confidence < CONFIDENCE_THRESHOLD:
            display_text = "Analizando..." # Ocultar predicción si es muy baja

        # Dibujar rectángulo de fondo para el texto (para que se lea mejor)
        cv2.rectangle(frame, (10, 10), (450, 60), (0, 0, 0), -1) 
        
        # Texto de predicción
        cv2.putText(frame, display_text, (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        # Texto de FPS (esquina inferior derecha)
        cv2.putText(frame, f"FPS: {int(fps)}", (500, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Mostrar ventana
        cv2.imshow('Detector de Residuos - IA', frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

if __name__ == "__main__":
    main()