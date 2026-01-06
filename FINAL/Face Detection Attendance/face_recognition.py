import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model

model = load_model("face_recognition_model.keras")
with open("results_map.pkl", "rb") as f:
    results_map = pickle.load(f)

cap = cv2.VideoCapture(0)

def preprocess(frame):
    img = cv2.resize(frame, (64, 64))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 32:
        preds = model.predict(preprocess(frame), verbose=0)[0]
        top10 = np.argsort(preds)[-10:][::-1]

        print("\nTOP 10 GUESSES")
        for i in range(10):
            ids = top10[i]
            name = results_map[ids]
            confidence = preds[ids] * 100
            print(f"{i+1}. {name} : {confidence:.2f} %")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
