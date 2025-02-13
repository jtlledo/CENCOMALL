import cv2

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
        
