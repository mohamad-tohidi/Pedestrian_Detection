# cython: language_level=3

# Import necessary Cython modules
cimport numpy as np
import numpy as np

# Rest of your imports...
import os
import supervision as sv
from ultralytics import YOLO
import cv2
import pygame
from shapely.geometry import Point, Polygon
import datetime
from datetime import datetime as dt

cdef double last_sound_played_at = 0  # Initialize to 0

cpdef np.ndarray[np.uint8_t, ndim=3] process_frame(np.ndarray[np.uint8_t, ndim=3] frame, _,
                                                  object model, object zone,
                                                  object box_annotator,
                                                  object zone_annotator,
                                                  object warning_sound,
                                                  int frame_count, int SKIP_RATE, int CLASS_ID):
    global last_sound_played_at

    if frame_count % SKIP_RATE == 0:
        results = model(frame, imgsz=384)[0]
        detections = sv.Detections.from_yolov8(results)
        detections = detections[detections.class_id == CLASS_ID]

        # Convert the polygon_coords to a Shapely Polygon
        polygon_shape = Polygon(zone.polygon)

        # Loop through each detection to check if it's inside the danger zone
        for bbox in detections.xyxy:
            x1, y1, x2, y2 = bbox
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            point = Point(x_center, y_center)

            if polygon_shape.contains(point):
                current_time = dt.now().timestamp()  # Use the renamed datetime class
                # Check if the sound was played in the last 10 seconds
                if current_time - last_sound_played_at > 10:
                    warning_sound.play()  # Play the preloaded warning sound
                    last_sound_played_at = current_time  # Update the timestamp
                    break

        zone.trigger(detections=detections)
        frame = box_annotator.annotate(scene=frame, detections=detections)

    frame = zone_annotator.annotate(scene=frame)

    frame_count += 1
    return frame
