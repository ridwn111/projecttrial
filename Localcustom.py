import argparse
from PIL import Image
import numpy as np
import torch
import cv2
import pyttsx3
import threading
import queue
import time
import os
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

model = None
camera = None  # Declare camera outside the functions
engine = pyttsx3.init()  # Initialize text-to-speech engine
voice_queue = queue.Queue()  # Queue for storing texts to be spoken
last_detected_objects = {}  # Dictionary to store the last detected objects and their last detection times
timeout_duration = 5  # Timeout duration in seconds

def init_camera(camera_index):
    global camera
    camera = cv2.VideoCapture(camera_index)  # Initialize camera feed

def detect_objects(frame):
    global model
    if model is None:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s', force_reload=True)
        model.eval()

    img = Image.fromarray(frame)
    results = model(img)  # Perform inference
    processed_frame = results.render()[0]  # Process the results
    
    # Get detected object labels
    labels = results.names

    # Get detected objects and their scores
    detected_objects = results.pred
    objects = []
    scores = []
    for obj in detected_objects[0]:
        label_index = int(obj[-1])  # Convert tensor to integer
        label = labels[label_index]
        confidence = obj[-2]
        if confidence > 0.5:  # Check confidence threshold
            objects.append(label)
            scores.append(confidence)
    return processed_frame, objects, scores

def voice_output():
    while True:
        text = voice_queue.get()  # Get text from the queue
        engine.say(text)
        engine.runAndWait()
        voice_queue.task_done()  # Mark the task as done after completion

def run_object_detection(camera_index=1):
    init_camera(camera_index)
    voice_thread = threading.Thread(target=voice_output)
    voice_thread.daemon = True
    voice_thread.start()  # Start voice thread
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            processed_frame, objects, scores = detect_objects(frame)
            cv2.imshow('Object Detection', processed_frame)

            current_time = time.time()
            # Enqueue objects for voice output if they are different from the last detected object or timeout has elapsed
            for obj, score in zip(objects, scores):
                last_detection_time = last_detected_objects.get(obj, 0)
                if current_time - last_detection_time > timeout_duration:
                    voice_queue.put(f"I see a {obj}")
                    last_detected_objects[obj] = current_time
                
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object detection using YOLOv5")
    parser.add_argument("--camera", type=int, default=1, help="Camera index (default is 1 for external camera)")
    args = parser.parse_args()
    
    run_object_detection(args.camera)
