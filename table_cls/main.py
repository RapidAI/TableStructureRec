import time

from pathlib import Path
import numpy as np
import onnxruntime
from PIL import Image

from .utils import InputType, LoadImage

cur_dir = Path(__file__).resolve().parent
table_cls_model_path = cur_dir / "models" / "table_cls.onnx"


class TableCls:
    def __init__(self, device="cpu"):
        providers = (
            ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        )
        self.table_cls = onnxruntime.InferenceSession(
            table_cls_model_path, providers=providers
        )
        self.inp_h = 224
        self.inp_w = 224
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.cls = {0: "wired", 1: "wireless"}
        self.load_img = LoadImage()

    def _preprocess(self, image):
        img = Image.fromarray(np.uint8(image))
        img = img.resize((self.inp_h, self.inp_w))
        img = np.array(img, dtype=np.float32) / 255.0
        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension, only one image
        return img

    def __call__(self, content: InputType):
        ss = time.perf_counter()
        img = self.load_img(content)
        img = self._preprocess(img)
        output = self.table_cls.run(None, {"input": img})
        predict = np.exp(output[0] - np.max(output[0], axis=1, keepdims=True))
        predict /= np.sum(predict, axis=1, keepdims=True)
        predict_cla = np.argmax(predict, axis=1)[0]
        table_elapse = time.perf_counter() - ss
        return self.cls[predict_cla], table_elapse
