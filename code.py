import cv2
import numpy as np

# Load the pre-trained MobileNetSSD model and the corresponding prototxt file
# Load the pre-trained MobileNetSSD model and the corresponding prototxt file
net = cv2.dnn.readNetFromCaffe(r"C:\Users\ADMIN\Downloads\deploy.prototxt", r"C:\Users\ADMIN\Downloads\mobilenetiter.caffemodel")

# Define the list of class labels MobileNetSSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
           "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
           "person", "smartphone", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Estimated real-world heights of objects in centimeters
KNOWN_HEIGHTS = {
    "person": 170,
    "bottle": 30,
    "car": 150,
    "bus": 300,
    "chair": 100,
    "diningtable": 75,
    "tvmonitor": 60,
    # Add more known heights for other classes as needed
}

# Colors for different classes (for bounding boxes)
Colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Focal length of the camera (this can be estimated through calibration)
FOCAL_LENGTH = 615 # Adjust this based on your camera

# Distance thresholds in centimeters
NEAR_DISTANCE_THRESHOLD = 50 # Alert if objects are closer than this distance

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the frame for detection
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Pass the blob through the network and obtain detections
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]

            if label in KNOWN_HEIGHTS:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Calculate perceived height of the object in pixels
                perceived_height = endY - startY

                # Estimate the distance from the camera to the object
                if perceived_height > 0:
                    distance = (KNOWN_HEIGHTS[label] * FOCAL_LENGTH) / perceived_height

                    # Draw the bounding box and label on the frame
                    display_text = "{}: {:.2f}cm".format(label, distance)
                    color = Colors[idx]
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    cv2.putText(frame, display_text, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Check if the object is too close and show alert
                    if distance < NEAR_DISTANCE_THRESHOLD:
                        cv2.putText(frame, "!! TOO CLOSE !!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

    # Display the output frame
    cv2.imshow("Real-time Object Detection and Distance Measurement", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()