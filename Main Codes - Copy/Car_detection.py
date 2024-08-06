import cv2

# Load the video capture
cap = cv2.VideoCapture(r'')

# Load the trained classifier
car_cascade = cv2.CascadeClassifier('cars.xml')

while True:
    # Read the frame
    _, frame = cap.read()

    # Convert the frame to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect cars in the grayscale frame
    cars = car_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around the detected cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # Display the result
    cv2.imshow('Car Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()
