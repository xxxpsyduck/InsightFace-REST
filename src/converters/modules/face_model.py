import math
import numpy as np
from numpy.linalg import norm
import cv2
from threading import Thread
from insightface.utils import face_align
from modules.face_align import FaceAligner
from modules.model_zoo.getter import get_model
from modules.imagedata import ImageData
import time
fa = FaceAligner()

device2ctx = {
    'cpu': -1,
    'cuda': 0
}


class Detector(Thread):
    def __init__(self, det_input_queue=None, det_output_queue=None, rec_input_queue=None, device: str = 'cuda', 
                det_name: str = 'retinaface_r50_v1', backend_name: str = 'trt', force_fp16: bool = False, max_size=[1280, 720]):
        super(Detector, self).__init__()
        self.det_input_queue = det_input_queue
        self.det_output_queue = det_output_queue
        self.rec_input_queue = rec_input_queue
        self.max_size = max_size
        self.retina = get_model(det_name, backend_name=backend_name, force_fp16=force_fp16,
                                im_size=self.max_size, root_dir='/models', download_model=False)
        self.retina.prepare(ctx_id=device2ctx[device], nms=0.4)

    def reproject_points(self, dets, scale: float):
        if scale != 1.0:
            dets = dets / scale
        return dets

    def run(self):
        while True:
            t0 = time.time()
            input_image = self.det_input_queue.get()
            t1 = time.time()
            print('det input queue get time:', t1 -t0)
            if input_image is not None:
                img = ImageData(input_image, max_size=self.max_size)
                img.resize_image(mode='pad')
                # self.det_output_queue.put(img)
                t2 = time.time()
                # print('resize time:', t2 - t1)
                bboxes, landmarks = self.retina.detect(img.transformed_image, threshold=0.9)
                t3 = time.time()
                # print('det model time', t3 - t2)
                boxes = bboxes[:, 0:4]
                face_data = {}
                if not isinstance(boxes, type(None)):
                    for i in range(len(boxes)):
                        bbox = self.reproject_points(boxes[i], img.scale_factor)
                        # cv2.rectangle(img.orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                        landmark = self.reproject_points(
                            landmarks[i], img.scale_factor)
                        # _crop = face_align.norm_crop(img.orig_image, landmark=landmark)
                        # face_data.append(_crop)
                        face_data[i] = {'bbox': bbox, 'landmark': landmark}
                t4 = time.time()
                # print('reproject point time:', t4 - t3)
                        # print(face_data)
                self.rec_input_queue.put(face_data)
                t5 = time.time()
                # print('rec input queue put time', t5 - t4)
                self.det_output_queue.put(img)
                t6 = time.time()
                # print('det output queue put time', t6 - t5)
                # print('total det:', t6 - t0)
            # print('det time:', time.time() - t0)


class Recognizer(Thread):
    def __init__(self, det_output_queue=None, rec_input_queue=None, rec_output_queue=None, rec_name: str = 'arcface_r100_v1', 
                 backend_name: str = 'trt', force_fp16: bool = False, device: str = 'cuda',
                 max_rec_batch_size: int = 10, embedding_index=None, user_index=None):
        super(Recognizer, self).__init__()
        self.rec_input_queue = rec_input_queue
        self.rec_output_queue = rec_output_queue
        self.det_output_queue = det_output_queue
        self.embedding_index = embedding_index
        self.user_index = user_index
        self.rec_model = get_model(rec_name, backend_name=backend_name, force_fp16=force_fp16,
                                   download_model=False, max_batch_size=max_rec_batch_size)
        self.rec_model.prepare(ctx_id=device2ctx[device])

    def run(self):
        while True:
            t0 = time.time()
            face_data = self.rec_input_queue.get()
            t1 = time.time()
            # print('rec input queue get time', t1 - t0)
            if face_data is not None:
                img = self.det_output_queue.get()
                t2 = time.time()
                # print('det output get time:', t2 - t1)
                crops = []
                for i, face in face_data.items():
                    landmark = np.array(face['landmark'])
                    _crop = face_align.norm_crop(img.orig_image, landmark=landmark)
                    # eyes = landmark[0:2]
                    # _crop = fa.align(img.orig_image, eyes)
                    crops.append(_crop)
                t3 = time.time()
                # print('align time:', t3 - t2)
                if len(crops) >= 1:                    
                    embeddings = self.rec_model.get_embeddings(crops)
                    t4 = time.time()
                    # print('model time:', t4 - t3)
                    for i, emb in enumerate(embeddings):
                        embedding = emb
                        embedding_norm = norm(embedding)
                        normed_embedding = embedding / embedding_norm
                        t5 = time.time()
                        # print('normalize time:', t5 - t4)
                        f_dists, f_ids = self.embedding_index.search(
                            normed_embedding.reshape(-1, 512).astype(np.float32), k=1)
                        f_dists = f_dists[0][0]
                        if f_dists > 0.5:
                            result_ids = f_ids[0][0]
                            result_user = self.user_index[result_ids]
                        # elif f_dists > 0.2:
                        #     result_user = 'checking'
                        else:
                            result_user = str(f_dists)
                        box = face_data[i]['bbox']
                        t6 = time.time()
                        # print('search time:', t6 - t5)
                        cv2.rectangle(img.orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                        cv2.putText(img.orig_image, result_user + ' ' + str(distance(box[0:2], box[2:4])), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        t7 = time.time()
                        # print('draw time:', t7 - t6)
                        # print('total rec:', t7 - t0)
                self.rec_output_queue.put(img.orig_image)        
            
def distance(p1, p2):
    distance = math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
    return distance