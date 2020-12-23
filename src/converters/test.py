import cv2
import numpy as np
import glob
from tqdm import tqdm
import faiss
import pickle
# from modules.face_model_org import FaceAnalysis
import imutils

# FA = FaceAnalysis(max_size=[1024,768])
# data = {}
# with open('embedding_low.pkl', 'wb') as f:
#     for path in tqdm(glob.glob('Faces' + '**/*/*')):
#         img = cv2.imread(path)
#         path = path.split('/')
#         face = FA.get(img)[0]
#         embedding = face.normed_embedding
#         name = path[-2]
#         data[name] = embedding
#     pickle.dump(data, f)
    

# with open('embedding.pkl', 'rb') as f:
#     user_index = []
#     data = pickle.load(f)
#     embedding_index = faiss.IndexFlatIP(512)
#     for key, value in data.items():
#         embedding_index.add(value.reshape(-1, 512).astype(np.float32))
#         user_index.append(key) 
#     print(user_index)


img = cv2.imread('test.jpg')
(_, encodedImage) = cv2.imencode(".jpg", img)
print(type(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n'))