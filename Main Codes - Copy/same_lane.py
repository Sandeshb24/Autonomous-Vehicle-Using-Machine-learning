import cv2

# Load pre-trained vehicle detection classifier (here, using Haar Cascade)
vehicle_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

# Capture video from a camera or a video file
cap = cv2.VideoCapture(r'C:\Users\Sandesh Bhat\Desktop\PROJECT\sim_code\Car_Detection_System-master\sample2.mp4')

# Set parameters for vehicle detection
min_size = (30, 30)
max_size = (300, 300)
scale_factor = 1.2
min_neighbors = 5

# Set threshold distance for warning (in pixels)
threshold_distance = 50

# Set warning message
warning_message = "Vehicle too close!"

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect vehicles in the frame using the pre-trained classifier
    vehicles = vehicle_cascade.detectMultiScale(gray, scaleFactor=scale_factor, 
                                                minNeighbors=min_neighbors, 
                                                minSize=min_size, maxSize=max_size)

    # Draw bounding boxes around the detected vehicles
    for (x, y, w, h) in vehicles:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Check if the vehicle is in our lane (here, checking if it's in the middle third of the frame)
        if x+w/2 > frame.shape[1]//3 and x+w/2 < 2*frame.shape[1]//3:
            # Calculate the distance between our vehicle and the detected vehicle
            distance = abs(y+h//2 - frame.shape[0]//2)

            # If the distance is less than the threshold distance, display warning message
            if distance < threshold_distance:
                cv2.putText(frame, warning_message, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Vehicle Detection', frame)

    # Wait for a key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the display window
cap.release()
cv2.destroyAllWindows()
