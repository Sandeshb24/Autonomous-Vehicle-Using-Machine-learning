import cv2
import numpy as np
import winsound

# Load the video stream from the camera
cap = cv2.VideoCapture(r'DEMO VIDEO\sample1_crop.mp4')

# Define the threshold values for obstacle detection
lower_threshold = (0, 0, 0)
upper_threshold = (100, 100, 100)

# Define the area of interest for obstacle detection
roi = (0, 200, 640, 480)

# Define the maximum speed of the vehicle
max_speed = 40

# Load the trained classifier
car_cascade = cv2.CascadeClassifier('cars.xml')

while True:
    # Read the current frame from the video stream
    ret, frame = cap.read()

    # Car and obstacle detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    cars = car_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # Lane detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    mask = np.zeros_like(edges)
    height, width = mask.shape
    polygon = np.array([[(0, height), (width, height), (width//2, height//2)]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(masked_edges, rho=6, theta=np.pi/60, threshold=160, minLineLength=40, maxLineGap=25)
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    # Obstacle detection
    x, y, w, h = roi
    frame_roi = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        obstacle_size = cv2.contourArea(largest_contour)
        brake_percentage = int((obstacle_size / (w * h)) * 100)
    else:
        brake_percentage = 0

    # Calculate the speed of the vehicle based on the brake percentage
    speed = max_speed - (max_speed * brake_percentage // 100)+5

    # Display the brake percentage and the speed on the frame
    brake_percentage=brake_percentage-5
   
        
    
    cv2.putText(combo_image, f"Brake: {brake_percentage}%", (1000, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combo_image, f"Speed: {speed} km/h", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if  brake_percentage> 55 and brake_percentage<=65 :
        frequency = 1000  # Set frequency to 1000 Hz
        duration = 40 # Set duration to 1000 milliseconds (1 second)
        winsound.Beep(frequency, duration)  # Produce the sound
        cv2.putText(combo_image, f"WARNING", (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
      

    elif brake_percentage> 66:
            frequency = 1200  # Set frequency to 1000 Hz
            duration = 80 # Set duration to 1000 milliseconds (1 second)
            winsound.Beep(frequency, duration)  # Produce the sound
            cv2.putText(combo_image, f"WARNING HARD BRAKING ", (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Car, Lane and Obstacle Detection", combo_image)

    # Check for key press and exit if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()