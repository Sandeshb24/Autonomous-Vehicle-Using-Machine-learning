import cv2
import numpy as np

# Load the video capture
cap = cv2.VideoCapture(r'DEMO VIDEO\sample1_crop.mp4')

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

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Create a mask for the region of interest
    mask = np.zeros_like(edges)
    height, width = mask.shape
    polygon = np.array([[(0, height), (width, height), (width//2, height//2)]], np.int32)
    cv2.fillPoly(mask, polygon, 255)

    # Apply the mask to the edges image
    masked_edges = cv2.bitwise_and(edges, mask)

    # Apply Hough transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, rho=6, theta=np.pi/60, threshold=160, minLineLength=40, maxLineGap=25)

    # Draw the detected lines on the original frame
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Combine the line and rectangle image with the original frame
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    # Display the result
    cv2.imshow('Car and Lane Detection', combo_image)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()
