import cv2
import numpy as np
import winsound

# Load the video file
cap = cv2.VideoCapture(r'C:\Users\Sandesh Bhat\Desktop\PROJECT\sim_code\Car_Detection_System-master\project_video.mp4')

# Define the lower and upper bounds for the color of the lane
lane_lower = np.array([0, 0, 100])
lane_upper = np.array([179, 50, 255])

# Define the lower and upper bounds for the color of the vehicles
vehicle_lower = np.array([0, 0, 0])
vehicle_upper = np.array([179, 255, 50])

# Loop through the frames in the video
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the frame to get the lane and vehicle masks
    lane_mask = cv2.inRange(hsv, lane_lower, lane_upper)
    vehicle_mask = cv2.inRange(hsv, vehicle_lower, vehicle_upper)

    # Find contours in the lane and vehicle masks
    lane_contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vehicle_contours, _ = cv2.findContours(vehicle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the vehicle contours and check if they are in the same lane as ours
    for contour in vehicle_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w // 2
        cy = y + h // 2

        # Check if the center of the vehicle contour is within the lane contour
        for lane_contour in lane_contours:
            if cv2.pointPolygonTest(lane_contour, (cx, cy), False) >= 0:
                # If the vehicle is in the same lane, trigger the warning sound
                winsound.PlaySound(r'C:\Users\Sandesh Bhat\Desktop\PROJECT\sim_code\Car_Detection_System-master\sound.mp3', winsound.SND_ASYNC)

    # Display the frame with the lane and vehicle contours
    cv2.drawContours(frame, lane_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(frame, vehicle_contours, -1, (0, 0, 255), 2)
    cv2.imshow('frame', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break
# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
