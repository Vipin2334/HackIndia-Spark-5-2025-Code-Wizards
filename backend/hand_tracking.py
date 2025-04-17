import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize Mediapipe for hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

# Initialize Mediapipe Hands module with parameters for hand detection and tracking
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Flip the frame for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB (Mediapipe expects RGB input)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        results = hands.process(rgb_frame)

        # If hands are detected, process them
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the landmarks and connections
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                          mp_drawing_styles.get_default_hand_landmarks_style(), 
                                          mp_drawing_styles.get_default_hand_connections_style())

                # Extract hand landmarks and flatten them to create feature vector
                landmarks = hand_landmarks.landmark
                hand_features = []

                for landmark in landmarks:
                    hand_features.append(landmark.x)
                    hand_features.append(landmark.y)
                    hand_features.append(landmark.z)

                # Convert to a numpy array and reshape it for the model
                hand_features = np.array(hand_features).reshape(1, -1)

                # Predict the letter using the trained model
                try:
                    prediction = model.predict(hand_features)
                    predicted_letter = prediction[0]

                    # Display the predicted letter on the frame
                    cv2.putText(frame, f"Prediction: {predicted_letter}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 255, 0), 2, cv2.LINE_AA)
                except Exception as e:
                    print(f"Prediction error: {e}")

        # Display the frame with predicted text
        cv2.imshow('Sign Language Detection', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
