import cv2
import os

# Ask for letter input (A-Z)
letter = input("Enter the letter you want to capture images for: ").upper()
SAVE_DIR = f"data/{letter}"

# Create folder if it doesn't exist
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Start capturing from webcam
cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show webcam feed
    cv2.imshow("Capture Window", frame)

    # Capture image when 'C' is pressed
    key = cv2.waitKey(1)
    if key & 0xFF == ord('c'):
        img_path = f"{SAVE_DIR}/img_{count}.jpg"
        cv2.imwrite(img_path, frame)
        print(f"Captured {img_path}")
        count += 1

    # Exit when 'Q' is pressed
    elif key & 0xFF == ord('q'):
        break

# Release webcam and close the window
cap.release()
cv2.destroyAllWindows()
