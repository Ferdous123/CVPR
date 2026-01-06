import cv2
import numpy as np
from tensorflow import keras
import sys

model = keras.models.load_model('mnist_handwritten_model.h5')
print("Model loaded\n")

def preprocess_mnist_style(frame):
    
    # Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # white digit on black
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        resized = cv2.resize(binary, (28, 28))
        normalized = resized.astype("float32") / 255.0
        return normalized.reshape(-1, 28*28), resized
    
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    digit = binary[y:y+h, x:x+w]
    
    target_size = 20
    
    if h > w:
        # Tall digit
        new_h = target_size
        new_w = int(w * target_size / h)
    else:
        # Wide digit
        new_w = target_size
        new_h = int(h * target_size / w)
    
    # Resize digit to target size
    if new_w > 0 and new_h > 0:
        resized_digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized_digit = digit
    
    # Create 28x28 BLACK canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)
    
    y_offset = (28 - resized_digit.shape[0]) // 2
    x_offset = (28 - resized_digit.shape[1]) // 2
    
    canvas[
        y_offset:y_offset + resized_digit.shape[0],
        x_offset:x_offset + resized_digit.shape[1]
    ] = resized_digit
    
    normalized = canvas.astype("float32") / 255.0
    
    flattened = normalized.reshape(-1, 28*28)
    
    return flattened, canvas

# Webcam setup
cap = cv2.VideoCapture(0)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

box_size = 280
x1 = (frame_width - box_size) // 2
y1 = (frame_height - box_size) // 2
x2 = x1 + box_size
y2 = y1 + box_size

current_prediction = None
confidence = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(frame, "Write digit here", (x1, y1-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Show prediction
    if current_prediction is not None:
        color = (0, 255, 0) if confidence > 80 else (0, 165, 255) if confidence > 50 else (0, 0, 255)
        pred_text = f"Digit: {current_prediction} ({confidence:.1f}%)"
        cv2.putText(frame, pred_text, (15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
    
    cv2.putText(frame, "SPACE=Predict | Q=Quit", (15, frame_height-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Webcam', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    
    elif key == ord(' '):
        roi = frame[y1:y2, x1:x2]
        input_data, processed = preprocess_mnist_style(roi)
        
        predictions = model.predict(input_data, verbose=0)
        predicted_digit = np.argmax(predictions[0])
        confidence = predictions[0][predicted_digit] * 100
        
        current_prediction = predicted_digit
        
        display = cv2.resize(processed, (280, 280), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Processed (28x28) - Should be ~20px digit', display)

cap.release()
cv2.destroyAllWindows()
print("Done\n")