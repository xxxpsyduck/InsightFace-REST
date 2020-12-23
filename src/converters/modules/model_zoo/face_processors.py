from .detectors.retinaface import RetinaFace
from .detectors.centerface import CenterFace



def arcface_r100_v1(model_path, backend, outputs):
    model = backend.Arcface(rec_name = model_path)
    return model


def genderage_v1(model_path, backend, outputs):
    model = backend.FaceGenderage(rec_name = model_path)
    return model




