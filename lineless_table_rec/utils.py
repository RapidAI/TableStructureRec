# -*- encoding: utf-8 -*-
import traceback
from io import BytesIO
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from PIL import Image, UnidentifiedImageError

root_dir = Path(__file__).resolve().parent
InputType = Union[str, np.ndarray, bytes, Path]


class OrtInferSession:
    def __init__(self, model_path: Union[str, Path], num_threads: int = -1):
        self.verify_exist(model_path)

        self.num_threads = num_threads
        self._init_sess_opt()

        cpu_ep = "CPUExecutionProvider"
        cpu_provider_options = {
            "arena_extend_strategy": "kSameAsRequested",
        }
        EP_list = [(cpu_ep, cpu_provider_options)]
        try:
            self.session = InferenceSession(
                str(model_path), sess_options=self.sess_opt, providers=EP_list
            )
        except TypeError:
            # 这里兼容ort 1.5.2
            self.session = InferenceSession(str(model_path), sess_options=self.sess_opt)

    def _init_sess_opt(self):
        self.sess_opt = SessionOptions()
        self.sess_opt.log_severity_level = 4
        self.sess_opt.enable_cpu_mem_arena = False

        if self.num_threads != -1:
            self.sess_opt.intra_op_num_threads = self.num_threads

        self.sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    def __call__(self, input_content: List[np.ndarray]) -> np.ndarray:
        input_dict = dict(zip(self.get_input_names(), input_content))
        try:
            return self.session.run(None, input_dict)
        except Exception as e:
            error_info = traceback.format_exc()
            raise ONNXRuntimeError(error_info) from e

    def get_input_names(
        self,
    ):
        return [v.name for v in self.session.get_inputs()]

    def get_output_name(self, output_idx=0):
        return self.session.get_outputs()[output_idx].name

    def get_metadata(self):
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        return meta_dict

    @staticmethod
    def verify_exist(model_path: Union[Path, str]):
        if not isinstance(model_path, Path):
            model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exist!")

        if not model_path.is_file():
            raise FileExistsError(f"{model_path} must be a file")


class ONNXRuntimeError(Exception):
    pass


class LoadImage:
    def __init__(
        self,
    ):
        pass

    def __call__(self, img: InputType) -> np.ndarray:
        if not isinstance(img, InputType.__args__):
            raise LoadImageError(
                f"The img type {type(img)} does not in {InputType.__args__}"
            )

        img = self.load_img(img)
        img = self.convert_img(img)
        return img

    def load_img(self, img: InputType) -> np.ndarray:
        if isinstance(img, (str, Path)):
            self.verify_exist(img)
            try:
                img = np.array(Image.open(img))
            except UnidentifiedImageError as e:
                raise LoadImageError(f"cannot identify image file {img}") from e
            return img

        if isinstance(img, bytes):
            img = np.array(Image.open(BytesIO(img)))
            return img

        if isinstance(img, np.ndarray):
            return img

        raise LoadImageError(f"{type(img)} is not supported!")

    def convert_img(self, img: np.ndarray):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.ndim == 3:
            channel = img.shape[2]
            if channel == 1:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            if channel == 2:
                return self.cvt_two_to_three(img)

            if channel == 4:
                return self.cvt_four_to_three(img)

            if channel == 3:
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            raise LoadImageError(
                f"The channel({channel}) of the img is not in [1, 2, 3, 4]"
            )

        raise LoadImageError(f"The ndim({img.ndim}) of the img is not in [2, 3]")

    @staticmethod
    def cvt_four_to_three(img: np.ndarray) -> np.ndarray:
        """RGBA → BGR"""
        r, g, b, a = cv2.split(img)
        new_img = cv2.merge((b, g, r))

        not_a = cv2.bitwise_not(a)
        not_a = cv2.cvtColor(not_a, cv2.COLOR_GRAY2BGR)

        new_img = cv2.bitwise_and(new_img, new_img, mask=a)
        new_img = cv2.add(new_img, not_a)
        return new_img

    @staticmethod
    def cvt_two_to_three(img: np.ndarray) -> np.ndarray:
        """gray + alpha → BGR"""
        img_gray = img[..., 0]
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        img_alpha = img[..., 1]
        not_a = cv2.bitwise_not(img_alpha)
        not_a = cv2.cvtColor(not_a, cv2.COLOR_GRAY2BGR)

        new_img = cv2.bitwise_and(img_bgr, img_bgr, mask=img_alpha)
        new_img = cv2.add(new_img, not_a)
        return new_img

    @staticmethod
    def verify_exist(file_path: Union[str, Path]):
        if not Path(file_path).exists():
            raise LoadImageError(f"{file_path} does not exist.")


class LoadImageError(Exception):
    pass
