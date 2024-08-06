import cv2
import numpy as np

# Load the video capture
cap = cv2.VideoCapture(r'')

while True:
    # Read the frame
    _, frame = cap.read()

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

    # Combine the line image with the original frame
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    # Display the result
    cv2.imshow('Lane Detection', combo_image)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()
