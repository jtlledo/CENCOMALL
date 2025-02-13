import os
from collections import defaultdict
import cv2
import numpy as np
import time
import json
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from utils.utils import initialize_video_capture, get_video_properties, initialize_video_writer

config_dir = os.path.join(os.path.dirname(__file__), "..", "configs", "config.json")

with open(config_dir, "r") as f:
    config = json.load(f)
    video_path = config["video_path"]
    model_path = config["model_path"]
    output_path = config["output_path"]

# Load the YOLO model
model = YOLO(model_path)

# Open the video file and get properties
cap = initialize_video_capture(video_path)
w, h, fps = get_video_properties(cap)

# Initialize video writer
out = initialize_video_writer(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, w, h)

# Define the central zone (rectangle)
zone_width, zone_height = 400, 300  # Size of the zone
zone_x1 = (w - zone_width) // 2  # Top-left x-coordinate
zone_y1 = (h - zone_height) // 2  # Top-left y-coordinate
zone_x2 = zone_x1 + zone_width  # Bottom-right x-coordinate
zone_y2 = zone_y1 + zone_height  # Bottom-right y-coordinate

# Initialize variables
prev_time = time.time()
last_positions = {}
people_in_zone = set()  # Track people currently in the zone
total_entered = 0  # Total people who entered the zone
total_exited = 0  # Total people who exited the zone

def is_point_in_zone(x, y):
    """Check if a point (x, y) is inside the central zone."""
    return zone_x1 <= x <= zone_x2 and zone_y1 <= y <= zone_y2

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    annotator = Annotator(frame, line_width=2)
    results = model.track(frame, persist=True)

    # Draw the central zone
    cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (0, 255, 0), 2)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        current_frame_people = set()  # Track people in the current frame

        for box, conf, class_id, track_id in zip(boxes, confs, class_ids, track_ids):
            # Check if the detected object is a person (class_id == 0)
            if int(class_id) == 0:  # 0 is typically the class_id for "person" in YOLO
                label = f"{model.names[int(class_id)]} {conf:.2f} ID: {track_id}"
                annotator.box_label(box, label, color=colors(track_id, True))

                # Calculate the centroid of the bounding box
                centroid_x = int((box[0] + box[2]) / 2)
                centroid_y = int((box[1] + box[3]) / 2)

                # Draw the centroid
                cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)  # Red dot

                # Check if the centroid is in the central zone
                if is_point_in_zone(centroid_x, centroid_y):
                    current_frame_people.add(track_id)  # Add to current frame people
                    if track_id not in people_in_zone:
                        total_entered += 1  # Person entered the zone
                    people_in_zone.add(track_id)  # Update people in the zone

        # Check for people who exited the zone
        exited_people = people_in_zone - current_frame_people
        total_exited += len(exited_people)
        people_in_zone = current_frame_people  # Update people in the zone

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Display statistics on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"In Zone: {len(people_in_zone)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Entered: {total_entered}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Exited: {total_exited}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    out.write(frame)
    cv2.imshow("object-detection-tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

out.release()
cap.release()
cv2.destroyAllWindows()