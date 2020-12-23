from threading import Thread
import cv2
import numpy as np
import time
class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self
        
    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


class Camera(Thread):
    def __init__(self, camera_queue, src1=0, width=1024, height=768):
        super(Camera, self).__init__()
        self.camera_queue = camera_queue
        self.stream1 = cv2.VideoCapture(src1)
        # self.stream2 = cv2.VideoCapture(src2, cv2.CAP_FFMPEG)
        # self.stream1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # self.stream1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # self.stream2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # self.stream2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def run(self):
        while True:
            (self.grabbed, self.frame1) = self.stream1.read()
            # (self.grabbed, self.frame2) = self.stream2.read()
            if self.grabbed == True:
                # frame = np.vstack((self.frame1, self.frame2))
                self.camera_queue.put(self.frame1)
