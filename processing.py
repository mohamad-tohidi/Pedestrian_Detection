# processing.py

import os
import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2
import pygame
from shapely.geometry import Point, Polygon
import datetime

pygame.mixer.init()
warning_sound = pygame.mixer.Sound('warning1.wav')

MODEL_PATH = 'yolov8n_openvino_model/'
CLASS_ID = 0
SKIP_RATE = 0
frame_count = 0

def process_frame(frame, _):
    global frame_count

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
                warning_sound.play()  # Play the preloaded warning sound instantly
                break

        zone.trigger(detections=detections)
        frame = box_annotator.annotate(scene=frame, detections=detections)

    frame = zone_annotator.annotate(scene=frame)

    frame_count += 1
    return frame

def process_video_with_annotations(video_path, polygon_coords):
    global model, zone, box_annotator, zone_annotator

    model = YOLO(MODEL_PATH)
    video_info = sv.VideoInfo.from_video_path(video_path)

    polygon = np.array(polygon_coords)
    zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)
    box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=2, text_scale=2)
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.white(), thickness=4, text_thickness=2, text_scale=2)

    # Get the output video path
    output_video_path = video_path.rsplit('.', 1)[0] + "-results.mp4"
    sv.process_video(source_path=video_path, target_path=output_video_path, callback=process_frame)

def process_webcam_with_annotations(polygon_coords):
    global model, zone, box_annotator, zone_annotator, frame_count

    # Initialize the YOLO model
    model = YOLO(MODEL_PATH)

    # Set webcam resolution
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        return
    frame_resolution_wh = (frame.shape[1], frame.shape[0])

    polygon = np.array(polygon_coords)
    zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=frame_resolution_wh)
    box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.white(), thickness=6, text_thickness=6, text_scale=4)

    # Create results directory if not exists
    if not os.path.exists("results"):
        os.makedirs("results")

    # Generate a unique filename based on current timestamp
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'results/webcam_output_{current_time}.avi'

    # Initialize VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_filename, fourcc, 20.0, frame_resolution_wh)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        processed_frame = process_frame(frame, None)

        # Write the processed frame to the output video
        out.write(processed_frame)

        # Show the processed frame
        cv2.imshow("Processed Webcam Feed", processed_frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()  # Release the VideoWriter object
    cv2.destroyAllWindows()
