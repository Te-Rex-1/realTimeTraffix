import cv2
from ultralytics import YOLO
from collections import defaultdict


# Load the YOLO model
model = YOLO('yolo11n.pt')

class_list = model.names  # List of class names

# Open the video file
cap = cv2.VideoCapture('Data/Video/3.mp4')

# Define line positions for counting
line_y_red = 298  # Red line position
line_y_blue = line_y_red + 100  # Blue line position

# Variables to store counting and tracking information
counted_ids_red_to_blue = set()
counted_ids_blue_to_red = set()

# Dictionaries to count objects by class for each direction
count_red_to_blue = defaultdict(int)  # Moving downwards
count_blue_to_red = defaultdict(int)  # Moving upwards

# State dictionaries to track which line was crossed first
crossed_red_first = {}
crossed_blue_first = {}

# Loop through video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        # Get the current timestamp in seconds from the video
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    # Run YOLO tracking on the frame
    results = model.track(frame, persist=True)

    # Ensure results are not empty
    if results[0].boxes.data is not None:
        # Get the detected boxes, their class indices, and track IDs
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu()

        # Draw the lines on the frame
        cv2.line(frame, (190, line_y_red), (850, line_y_red), (0, 0, 255), 3)
        cv2.putText(frame, 'Red Line', (190, line_y_red - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

        cv2.line(frame, (27, line_y_blue), (960, line_y_blue), (255, 0, 0), 3)
        cv2.putText(frame, 'Blue Line', (27, line_y_blue - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

        # Loop through each detected object
        for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
            x1, y1, x2, y2 = map(int, box)

            cx = (x1 + x2) // 2  # Calculate the center point
            cy = (y1 + y2) // 2

            # Get the class name using the class index
            class_name = class_list[class_idx]

            # Draw a dot at the center and display the tracking ID and class name
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Check if the object crosses the red line
            if line_y_red - 5 <= cy <= line_y_red + 5:
                # Record that the object crossed the red line
                if track_id not in crossed_red_first:
                    crossed_red_first[track_id] = True

            # Check if the object crosses the blue line
            if line_y_blue - 5 <= cy <= line_y_blue + 5:
                # Record that the object crossed the blue line
                if track_id not in crossed_blue_first:
                    crossed_blue_first[track_id] = True

            # Counting logic for downward direction (red -> blue)
            if track_id in crossed_red_first and track_id not in counted_ids_red_to_blue:
                if line_y_blue - 5 <= cy <= line_y_blue + 5:
                    counted_ids_red_to_blue.add(track_id)
                    count_red_to_blue[class_name] += 1

            # Counting logic for upward direction (blue -> red)
            if track_id in crossed_blue_first and track_id not in counted_ids_blue_to_red:
                if line_y_red - 5 <= cy <= line_y_red + 5:
                    counted_ids_blue_to_red.add(track_id)
                    count_blue_to_red[class_name] += 1

    # Display the counts on the frame
    y_offset = 30
    for class_name, count in count_red_to_blue.items():
        cv2.putText(frame, f'{class_name} (Down): {count}', (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        y_offset += 30

    y_offset += 20  # Add spacing for upward counts
    for class_name, count in count_blue_to_red.items():
        cv2.putText(frame, f'{class_name} (Up): {count}', (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        y_offset += 30

    # Show the frame
    cv2.imshow("YOLO Object Tracking & Counting", frame)

    # Exit loop if 'ESC' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()