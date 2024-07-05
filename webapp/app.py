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
from flask import Flask, render_template, Response

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)

model = None
camera = None
engine = pyttsx3.init()
voice_queue = queue.Queue()
last_detected_objects = {}
timeout_duration = 10

def init_camera(camera_index):
    global camera
    camera = cv2.VideoCapture(camera_index)

def detect_objects(frame):
    global model
    if model is None:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s', force_reload=True)
        model.eval()

    img = Image.fromarray(frame)
    results = model(img)
    processed_frame = results.render()[0]
    
    labels = results.names
    detected_objects = results.pred
    objects = []
    scores = []
    for obj in detected_objects[0]:
        label_index = int(obj[-1])
        label = labels[label_index]
        confidence = obj[-2]
        if confidence > 0.5:
            objects.append(label)
            scores.append(confidence)
    return processed_frame, objects, scores

def voice_output():
    while True:
        text = voice_queue.get()
        engine.say(text)
        engine.runAndWait()
        voice_queue.task_done()

def generate_frames():
    init_camera(0)
    voice_thread = threading.Thread(target=voice_output)
    voice_thread.daemon = True
    voice_thread.start()
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            processed_frame, objects, scores = detect_objects(frame)
            current_time = time.time()
            for obj, score in zip(objects, scores):
                last_detection_time = last_detected_objects.get(obj, 0)
                if current_time - last_detection_time > timeout_duration:
                    voice_queue.put(f"I see a {obj}")
                    last_detected_objects[obj] = current_time

            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
