import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .utils import InputType, LoadImage, OrtInferSession

cur_dir = Path(__file__).resolve().parent
q_cls_model_path = cur_dir / "models" / "table_cls.onnx"
yolo_cls_model_path = cur_dir / "models" / "yolo_cls.onnx"
yolo_cls_x_model_path = cur_dir / "models" / "yolo_cls_x.onnx"


class TableCls:
    def __init__(self, model_type="yolo", model_path=yolo_cls_model_path):
        if model_type == "yolo":
            self.table_engine = YoloCls(model_path)
        elif model_type == "yolox":
            self.table_engine = YoloCls(yolo_cls_x_model_path)
        else:
            model_path = q_cls_model_path
            self.table_engine = QanythingCls(model_path)
        self.load_img = LoadImage()

    def __call__(self, content: InputType):
        ss = time.perf_counter()
        img = self.load_img(content)
        img = self.table_engine.preprocess(img)
        predict_cla = self.table_engine([img])
        table_elapse = time.perf_counter() - ss
        return predict_cla, table_elapse


class QanythingCls:
    def __init__(self, model_path):
        self.table_cls = OrtInferSession(model_path)
        self.inp_h = 224
        self.inp_w = 224
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.cls = {0: "wired", 1: "wireless"}

    def preprocess(self, img):
        img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.stack((img,) * 3, axis=-1)
        img = Image.fromarray(np.uint8(img))
        img = img.resize((self.inp_h, self.inp_w))
        img = np.array(img, dtype=np.float32) / 255.0
        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension, only one image
        return img

    def __call__(self, img):
        output = self.table_cls(img)
        predict = np.exp(output[0] - np.max(output[0], axis=1, keepdims=True))
        predict /= np.sum(predict, axis=1, keepdims=True)
        predict_cla = np.argmax(predict, axis=1)[0]
        return self.cls[predict_cla]


class YoloCls:
    def __init__(self, model_path):
        self.table_cls = OrtInferSession(model_path)
        self.cls = {0: "wireless", 1: "wired"}

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = np.array(img, dtype=np.float32) / 255
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension, only one image
        return img

    def __call__(self, img):
        output = self.table_cls(img)
        predict_cla = np.argmax(output[0], axis=1)[0]
        return self.cls[predict_cla]
