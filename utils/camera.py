import cv2
from PIL import Image

def capture_image_pil():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Could not open webcam.")
    
    ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()

    if not ret:
        raise IOError("Could not read frame from webcam.")
    
    # Convert from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_frame)
    
    return pil_image