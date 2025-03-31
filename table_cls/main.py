import time
from enum import Enum
from pathlib import Path
from typing import Union, Dict

import cv2
import numpy as np
from PIL import Image

from .utils.download_model import DownloadModel
from .utils.utils import InputType, LoadImage, OrtInferSession, resize_and_center_crop


class ModelType(Enum):
    YOLO_CLS_X = "yolox"
    YOLO_CLS = "yolo"
    PADDLE_CLS = "paddle"
    Q_CLS = "q"


ROOT_URL = "https://www.modelscope.cn/models/RapidAI/RapidTable/resolve/master/"
KEY_TO_MODEL_URL = {
    ModelType.YOLO_CLS_X.value: f"{ROOT_URL}/table_cls/yolo_cls_x.onnx",
    ModelType.YOLO_CLS.value: f"{ROOT_URL}/table_cls/yolo_cls.onnx",
    ModelType.PADDLE_CLS.value: f"{ROOT_URL}/table_cls/paddle_cls.onnx",
    ModelType.Q_CLS.value: f"{ROOT_URL}/table_cls/q_cls.onnx",
}


class TableCls:
    def __init__(self, model_type=ModelType.YOLO_CLS.value, model_path=None):
        model_path = self.get_model_path(model_type, model_path)
        if model_type == ModelType.YOLO_CLS.value:
            self.table_engine = YoloCls(model_path)
        elif model_type == ModelType.YOLO_CLS_X.value:
            self.table_engine = YoloCls(model_path)
        elif model_type == ModelType.PADDLE_CLS.value:
            self.table_engine = PaddleCls(model_path)
        else:
            self.table_engine = QanythingCls(model_path)
        self.load_img = LoadImage()

    def __call__(self, content: InputType):
        ss = time.perf_counter()
        img = self.load_img(content)
        img = self.table_engine.preprocess(img)
        predict_cla = self.table_engine([img])
        table_elapse = time.perf_counter() - ss
        return predict_cla, table_elapse

    @staticmethod
    def get_model_path(
        model_type: str, model_path: Union[str, Path, None]
    ) -> Union[str, Dict[str, str]]:
        if model_path is not None:
            return model_path

        model_url = KEY_TO_MODEL_URL.get(model_type, None)
        if isinstance(model_url, str):
            model_path = DownloadModel.download(model_url)
            return model_path

        if isinstance(model_url, dict):
            model_paths = {}
            for k, url in model_url.items():
                model_paths[k] = DownloadModel.download(
                    url, save_model_name=f"{model_type}_{Path(url).name}"
                )
            return model_paths

        raise ValueError(f"Model URL: {type(model_url)} is not between str and dict.")


class PaddleCls:
    def __init__(self, model_path):
        self.table_cls = OrtInferSession(model_path)
        self.inp_h = 224
        self.inp_w = 224
        self.resize_short = 256
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.cls = {0: "wired", 1: "wireless"}

    def preprocess(self, img):
        # short resize
        img_h, img_w = img.shape[:2]
        percent = float(self.resize_short) / min(img_w, img_h)
        w = int(round(img_w * percent))
        h = int(round(img_h * percent))
        img = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_LANCZOS4)
        # center crop
        img_h, img_w = img.shape[:2]
        w_start = (img_w - self.inp_w) // 2
        h_start = (img_h - self.inp_h) // 2
        w_end = w_start + self.inp_w
        h_end = h_start + self.inp_h
        img = img[h_start:h_end, w_start:w_end, :]
        # normalize
        img = np.array(img, dtype=np.float32) / 255.0
        img -= self.mean
        img /= self.std
        # HWC to CHW
        img = img.transpose(2, 0, 1)
        # Add batch dimension, only one image
        img = np.expand_dims(img, axis=0)
        return img

    def __call__(self, img):
        pred_output = self.table_cls(img)[0]
        pred_idxs = list(np.argmax(pred_output, axis=1))
        predict_cla = max(set(pred_idxs), key=pred_idxs.count)
        return self.cls[predict_cla]


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
        img = resize_and_center_crop(img, 640)
        img = np.array(img, dtype=np.float32) / 255
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension, only one image
        return img

    def __call__(self, img):
        output = self.table_cls(img)
        predict_cla = np.argmax(output[0], axis=1)[0]
        return self.cls[predict_cla]
