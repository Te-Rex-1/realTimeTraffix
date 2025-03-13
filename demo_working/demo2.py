
import cv2
import time
import math
import matplotlib.pyplot as plt
from collections import defaultdict
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('yolo11n.pt')
class_list = model.names  # List of class names

# Open video file
cap = cv2.VideoCapture(r'C:\Users\tar30\PycharmProjects\PythonProject3\Data\Video\3.mp4')


# Dictionaries to store timestamps
entry_times = {}  # Dictionary to store entry times for each track_id
exit_times = {}   # Dictionary to store exit times for each track_id
calculateSpeed = {}  # Dictionary to store calculated speeds

displayed_speeds = {}   # To persist speed display

# Define line positions for counting
line_y_track1 = 298  # Red line position
line_y_track2 = line_y_track1 + 100  # Blue line position

# Variables to store counting and tracking information
counted_ids_red_to_blue = set()
counted_ids_blue_to_red = set()

# Dictionaries to count objects by class for each direction
count_red_to_blue = defaultdict(int)  # Moving downwards
count_blue_to_red = defaultdict(int)  # Moving upwards

# State dictionaries to track which line was crossed first
crossed_red_first = {}
crossed_blue_first = {}

# Line positions for detection
line_y_track1 = 298
line_y_track2 = line_y_track1 + 100

# Store data for plotting
frame_times = []
vehicle_counts = []
average_speeds = []

# Set up real-time plotting
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

def update_plot():
    """ Updates the real-time graph. """
    ax1.clear()
    ax2.clear()

    ax1.plot(frame_times, vehicle_counts, color='b', marker='o', linestyle='-')
    ax1.set_title("Number of Vehicles Over Time")
    ax1.set_xlabel("Frame Count")
    ax1.set_ylabel("Number of Vehicles")

    ax2.plot(frame_times, average_speeds, color='r', marker='s', linestyle='-')
    ax2.set_title("Average Speed Over Time")
    ax2.set_xlabel("Frame Count")
    ax2.set_ylabel("Speed (KM/H)")

    plt.pause(0.01)  # Short pause for real-time updates

def calculate_pixel_distance(x1, y1, x2, y2):
    """ Calculates the Euclidean distance in pixels. """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_speed(time_diff, pixel_distance, pixels_per_meter=10):
    """ Converts pixel movement into speed (KM/H). """
    real_distance = pixel_distance / pixels_per_meter  # Convert to meters
    speed_mps = real_distance / time_diff  # Speed in meters per second
    speed_kph = speed_mps * 3.6  # Convert to KM/H
    return round(speed_kph, 2)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)

    if results[0].boxes.data is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, class_idx in zip(boxes, track_ids, class_indices):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            # Get the class name using the class index
            class_name = class_list[class_idx]
            # Draw bounding boxes and ID
            color = (0, 255, 0) if track_id in calculateSpeed else (0, 0, 255)  # Green for tracked, Red for new
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Check if object crosses the first line
            if line_y_track1 - 5 <= cy <= line_y_track1 + 5:
                if track_id not in crossed_red_first:
                    crossed_red_first[track_id] = True
                    entry_times[track_id] = time.time()

            # Check if object crosses the second line
            if line_y_track2 - 5 <= cy <= line_y_track2 + 5:
                if track_id not in crossed_blue_first:
                    crossed_blue_first[track_id] = True
                    exit_times[track_id] = time.time()

                    # Calculate speed
                    if track_id in entry_times:
                        time_diff = exit_times[track_id] - entry_times[track_id]
                        pixel_distance = calculate_pixel_distance(cx, line_y_track1, cx, line_y_track2)
                        calculateSpeed[track_id] = calculate_speed(time_diff, pixel_distance)


                        # Counting logic for downward direction (red -> blue)
                        if track_id in crossed_red_first and track_id not in counted_ids_red_to_blue:
                            if line_y_track2 - 5 <= cy <= line_y_track2 + 5:
                                counted_ids_red_to_blue.add(track_id)
                                count_red_to_blue[class_name] += 1
                                exit_times[track_id] = time.time()
                                calculateSpeed[track_id] = speed.calculate_speed(
                                    exit_times[track_id] - entry_times[track_id])
                                displayed_speeds[track_id] = calculateSpeed[track_id]  # Store for display

                        # Counting logic for upward direction (blue -> red)
                        if track_id in crossed_blue_first and track_id not in counted_ids_blue_to_red:
                            if line_y_track1 - 5 <= cy <= line_y_track1 + 5:
                                counted_ids_blue_to_red.add(track_id)
                                count_blue_to_red[class_name] += 1
                                exit_times[track_id] = time.time()
                                calculateSpeed[track_id] = speed.calculate_speed(
                                    exit_times[track_id] - entry_times[track_id])
                                displayed_speeds[track_id] = calculateSpeed[track_id]  # Store for display


        speed = calculateSpeed[track_id]
        cv2.putText(frame, f"Speed: {speed:.2f} KM/H", (x1, y2 + 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        # Update real-time graph data
        frame_count += 1
        frame_times.append(frame_count)
        vehicle_counts.append(len(calculateSpeed))
        avg_speed = sum(calculateSpeed.values()) / len(calculateSpeed) if calculateSpeed else 0
        average_speeds.append(avg_speed)

        update_plot()

    # Draw tracking lines
    cv2.line(frame, (190, line_y_track1), (850, line_y_track1), (0, 0, 255), 3)
    cv2.putText(frame, 'Track 1', (190, line_y_track1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)

    cv2.line(frame, (27, line_y_track2), (960, line_y_track2), (255, 0, 0), 3)
    cv2.putText(frame, 'Track 2', (27, line_y_track2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)

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
    cv2.imshow("YOLO Vehicle Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
