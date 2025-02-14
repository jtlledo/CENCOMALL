import cv2
import numpy as np

def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    return cap

def get_video_properties(cap):
    return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FPS)))

def initialize_video_writer(output_path, fourcc, fps, width, height):
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def save_logs(logs_path, data):
    with open(logs_path, "a") as f:
        # Extraemos los datos del diccionario y los guardamos en el archivo
        for key, value in data.items():
            f.write(f"{key}: {value}\n")

def detect_shoplifting(frame, model, confidence_threshold=0.8):
    """
    Detecta si una persona está robando en el frame utilizando el modelo de detección de robo.
    
    Args:
        frame (numpy.ndarray): El frame de video.
        model (YOLO): El modelo YOLO para detección de robo.
        confidence_threshold (float): Umbral de confianza para considerar una detección válida.
    
    Returns:
        numpy.ndarray: El frame con las anotaciones de detección de robo.
    """

    # Colores para diferenciar a las personas
    ROBBERY_COLOR = (0, 0, 255)  # Rojo para personas robando
    NORMAL_COLOR = (0, 255, 0)    # Verde para personas normales

    # Definir estados de detección
    shoplifting_status = "Robando"
    not_shoplifting_status = "No robando"
    result = model.predict(frame)
    cc_data = np.array(result[0].boxes.data)

    if len(cc_data) != 0:
        xywh = np.array(result[0].boxes.xywh).astype("int32")
        xyxy = np.array(result[0].boxes.xyxy).astype("int32")
        
        for (x1, y1, x2, y2), (_, _, _, _), (_, _, _, _, conf, clas) in zip(xyxy, xywh, cc_data):
            if conf >= confidence_threshold:  # Solo considerar detecciones con alta confianza
                if clas == 1:  # Clase 1: Robo
                    color = ROBBERY_COLOR  # Rojo para personas robando
                    status = shoplifting_status
                else:  # Clase 0: No robo
                    color = NORMAL_COLOR  # Verde para personas normales
                    status = not_shoplifting_status

                # Dibujar el cuadro alrededor de la persona
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Mostrar la confianza como texto
                text = f"{status} {conf * 100:.2f}%"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame
        
