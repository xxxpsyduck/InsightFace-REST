import time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
import cv2
import faiss
import pickle
import numpy as np
from queue import Queue
from threading import Thread
from modules.face_model import Detector, Recognizer
from camera import Camera

templates = Jinja2Templates(directory="templates")

det_input_queue = Queue()
det_output_queue = Queue()
rec_input_queue = Queue()
rec_output_queue = Queue()
out_put_frame_queue = Queue()
cam_queue = Queue()

app = FastAPI()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(out_put_frame_queue.get(), media_type='multipart/x-mixed-replace; boundary=frame')


@app.get("/stream")
def index(request: Request):
    return templates.TemplateResponse('index.html', {"request": request})

with open('embedding_low.pkl', 'rb') as f:
    user_index = []
    data = pickle.load(f)
    embedding_index = faiss.IndexFlatIP(512)
    for key, value in data.items():
        embedding_index.add(value.reshape(-1, 512).astype(np.float32))
        user_index.append(key)


class Streaming(Thread):
    def __init__(self, rec_output_queue=None):
        super(Streaming, self).__init__()
        self.rec_output_queue = rec_output_queue

    def run(self):
        while True:
            output_frame = self.rec_output_queue.get()
            if output_frame is not None:
                (_, encodedImage) = cv2.imencode(".jpg", output_frame)
                # yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
                # print(type(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n'))
                out_put_frame_queue.put(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

detector = Detector(det_input_queue=det_input_queue,
                    det_output_queue=det_output_queue, rec_input_queue=rec_input_queue)
recognizer = Recognizer(det_output_queue=det_output_queue, rec_input_queue=rec_input_queue, rec_output_queue=rec_output_queue,
                        embedding_index=embedding_index, user_index=user_index)
# cap = Camera(src1='rtsp://admin:phanpc@Edso@10.0.0.31/onvif/profile1/media.smp', src2='rtsp://admin:phanpc@Edso@10.0.0.32/onvif/profile1/media.smp', camera_queue=cam_queue)
cap = Camera(src1='rtsp://admin:admin@1357@10.0.0.29', camera_queue=cam_queue)
streaming = Streaming(rec_output_queue)

# def stream():
cap.start()    
detector.start()
recognizer.start()
streaming.start()
while True:
    frame = cam_queue.get()
    det_input_queue.put(frame)



