import cv2
import os
import numpy as np
import mediapipe as mp
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Initialize Mediapipe
mp_hands = mp.solutions.hands

# Prepare training data
X = []  # features
y = []  # labels

# Directory where images are stored
data_dir = 'data'

# Process each letter folder (A-Z)
for label in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, label)
    if os.path.isdir(folder_path):
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                continue

            # Convert to RGB for Mediapipe
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detect hand landmarks
            with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
                results = hands.process(rgb_img)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = hand_landmarks.landmark
                        features = [landmark.x for landmark in landmarks] + [landmark.y for landmark in landmarks] + [landmark.z for landmark in landmarks]
                        X.append(features)
                        y.append(label)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Train a KNeighbors Classifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model training complete and saved to model.pkl")
