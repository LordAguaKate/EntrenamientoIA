import cv2
import numpy as np
import time
import platform
import sys
from pathlib import Path

# --- GESTIÓN DE IMPORTACIONES (Compatibilidad Cross-Platform) ---
# Intenta usar tflite_runtime (ligero, ideal para Linux/RPi)
# Si no existe, usa tensorflow completo (ideal para Windows/Dev)
try:
    import tflite_runtime.interpreter as tflite
    print("Usando tflite_runtime (Modo Ligero)")
except ImportError:
    try:
        import tensorflow.lite as tflite
        print("Usando tensorflow.lite (Modo Completo)")
    except ImportError:
        print("Error: No se encontró 'tflite_runtime' ni 'tensorflow'.")
        print("Instala uno: pip install tflite-runtime OR pip install tensorflow")
        sys.exit(1)

# --- CONFIGURACIÓN ---
BASE_DIR = Path(__file__).resolve().parent.parent 
MODEL_PATH = BASE_DIR / "models" / "modelo_residuos_rpi.tflite"
CONFIDENCE_THRESHOLD = 0.50

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
        if not model_path.exists():
            print(f"Error: El archivo del modelo no existe en {model_path}")
            sys.exit(1)

        try:
            self.interpreter = tflite.Interpreter(model_path=str(model_path))
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_shape = self.input_details[0]['shape'][1:3] # (224, 224)
            print("Modelo cargado exitosamente.")
        except Exception as e:
            print(f"Error crítico cargando el modelo: {e}")
            sys.exit(1)

    def predict(self, frame):
        resized_frame = cv2.resize(frame, (self.input_shape[0], self.input_shape[1]))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(rgb_frame, axis=0).astype(np.float32)
        input_data = input_data / 255.0

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        idx = np.argmax(output_data)
        confidence = output_data[idx]
        
        return CLASSES[idx], confidence

def get_camera():
    """Configura la cámara dependiendo del Sistema Operativo"""
    system_os = platform.system()
    
    # En Windows, cv2.CAP_DSHOW suele ser más rápido y evita advertencias
    if system_os == 'Windows':
        print("Detectado Windows: Usando backend DirectShow...")
        return cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # En Linux (Fedora/Ubuntu/RPi), el backend por defecto suele ir bien (V4L2)
    print(f"Detectado {system_os}: Usando backend por defecto...")
    return cv2.VideoCapture(0)

def main():
    classifier = WasteClassifier(MODEL_PATH)
    cap = get_camera()

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    # Configurar ventana redimensionable (ayuda en algunos gestores de ventanas de Linux)
    cv2.namedWindow('Detector de Residuos - IA', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detector de Residuos - IA', 800, 600)

    print("--- INICIANDO SISTEMA ---")
    print("Presiona 'q' en la ventana o Ctrl+C en la terminal para salir.")

    prev_frame_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error capturando frame.")
                break

            label, confidence = classifier.predict(frame)

            # FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
            prev_frame_time = new_frame_time

            # UI
            color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 0, 255)
            display_text = f"{label}: {confidence*100:.1f}%"
            if confidence < CONFIDENCE_THRESHOLD:
                display_text = "Analizando..."

            cv2.rectangle(frame, (10, 10), (450, 60), (0, 0, 0), -1) 
            cv2.putText(frame, display_text, (20, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 450), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow('Detector de Residuos - IA', frame)

            # Salida controlada con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Salida solicitada por usuario (tecla 'q').")
                break

    except KeyboardInterrupt:
        print("\nPrograma detenido manualmente (Ctrl+C).")
    
    finally:
        # Este bloque SIEMPRE se ejecuta, haya error o no.
        print("Liberando recursos y cerrando...")
        cap.release()
        cv2.destroyAllWindows()
        # Pequeña pausa para asegurar que las ventanas gráficas se cierren en Linux
        cv2.waitKey(1)
        print("Finalizado correctamente.")

if __name__ == "__main__":
    main()