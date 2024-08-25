import numpy as np
import cv2
from flask import Flask, Response, request
from vidgear.gears import CamGear
import time
from typing import Union
import grpc
import os
import sys
sys.path.insert(0, './protos')
import yolo_pb2
import yolo_pb2_grpc
from queue import Queue
from threading import Thread

app = Flask(__name__)

yolo_api = os.environ.get("YOLO_API", "localhost:50051")
def detect_with_yolo(frame: np.ndarray) -> np.ndarray:
    with grpc.insecure_channel(yolo_api) as channel:
        yolo_stub = yolo_pb2_grpc.YoloStub(channel)

        # Encode the frame as JPEG
        _, frame_encoded = cv2.imencode('.jpg', frame)

        # Create a request to send to YOLO
        request = yolo_pb2.Image(data=frame_encoded.tobytes())

        try:
            # Send the frame to YOLO for detection
            response = yolo_stub.Track(request)

            # Check for errors in the response (assuming there's a status field)
            if response.metadata.status != "OK":
                print(f"YOLO detection error: {response.metadata.message}")
                return frame  # Return the original frame in case of error

            # Decode the received image from YOLO response
            nparr = np.frombuffer(response.data, np.uint8)
            output_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return output_frame

        except grpc.RpcError as e:
            print(f"gRPC error: {e.code()}: {e.details()}")
            return frame 


# options = {"STREAM_RESOLUTION": "480p"}
# def feed(url: Union[str, int]):
#     # Initialize video stream using CamGear
#     cap = CamGear(source=url, stream_mode=True, **options).start()
    
#     while True:
#         # Read a frame from the video stream
#         frame = cap.read()
#         if frame is None:
#             break

#         # Encode the frame as JPEG
#         ret, jpeg = cv2.imencode('.jpg', frame)
#         if not ret:
#             continue

#         # Yield the frame in the appropriate format for streaming
#         yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

#         time.sleep(0.05)

options = {"STREAM_RESOLUTION": "480p"}
def feed_with_yolo(source: Union[str, int]):
    # Initialize video stream using CamGear
    cap = CamGear(source=source, stream_mode=True, **options).start()

    while True:
        input_frame = cap.read()
        if input_frame is None:
            break

        # Send the frame to YOLO for detection
        output_frame = detect_with_yolo(input_frame)

        # Encode the detected frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', output_frame)
        if not ret:
            continue

        # Yield the frame in the appropriate format for streaming
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        time.sleep(0.05)


@app.route("/", methods=["GET"])
def feedRoute():
    url = request.args.get("url", default="https://www.youtube.com/watch?v=dQw4w9WgXcQ", type=str)
    
    if url == "0":
        url = 0
    else:
        # Ensure the URL is valid and starts with 'http' or 'rtsp' for streaming
        if not url.startswith(('http://', 'https://', 'rtsp://')):
            return "Invalid URL. Please provide a valid camera URL or '0' for the default webcam.", 400

    return Response(
        feed_with_yolo(url),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
