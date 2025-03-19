# -*- encoding: utf-8 -*-
import os
import platform
import traceback
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import List, Union, Dict, Any, Tuple, Optional

import cv2
import numpy as np
from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
    get_available_providers,
    get_device,
)
from PIL import Image, UnidentifiedImageError

from .logger import get_logger

root_dir = Path(__file__).resolve().parent
InputType = Union[str, np.ndarray, bytes, Path, Image.Image]


class EP(Enum):
    CPU_EP = "CPUExecutionProvider"
    CUDA_EP = "CUDAExecutionProvider"
    DIRECTML_EP = "DmlExecutionProvider"


class OrtInferSession:
    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger("OrtInferSession")

        model_path = config.get("model_path", None)
        self._verify_model(model_path)

        self.cfg_use_cuda = config.get("use_cuda", None)
        self.cfg_use_dml = config.get("use_dml", None)

        self.had_providers: List[str] = get_available_providers()
        EP_list = self._get_ep_list()

        sess_opt = self._init_sess_opts(config)
        self.session = InferenceSession(
            model_path,
            sess_options=sess_opt,
            providers=EP_list,
        )
        self._verify_providers()

    @staticmethod
    def _init_sess_opts(config: Dict[str, Any]) -> SessionOptions:
        sess_opt = SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = False
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        cpu_nums = os.cpu_count()
        intra_op_num_threads = config.get("intra_op_num_threads", -1)
        if intra_op_num_threads != -1 and 1 <= intra_op_num_threads <= cpu_nums:
            sess_opt.intra_op_num_threads = intra_op_num_threads

        inter_op_num_threads = config.get("inter_op_num_threads", -1)
        if inter_op_num_threads != -1 and 1 <= inter_op_num_threads <= cpu_nums:
            sess_opt.inter_op_num_threads = inter_op_num_threads

        return sess_opt

    def get_metadata(self, key: str = "character") -> list:
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        content_list = meta_dict[key].splitlines()
        return content_list

    def _get_ep_list(self) -> List[Tuple[str, Dict[str, Any]]]:
        cpu_provider_opts = {
            "arena_extend_strategy": "kSameAsRequested",
        }
        EP_list = [(EP.CPU_EP.value, cpu_provider_opts)]

        cuda_provider_opts = {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        }
        self.use_cuda = self._check_cuda()
        if self.use_cuda:
            EP_list.insert(0, (EP.CUDA_EP.value, cuda_provider_opts))

        self.use_directml = self._check_dml()
        if self.use_directml:
            self.logger.info(
                "Windows 10 or above detected, try to use DirectML as primary provider"
            )
            directml_options = (
                cuda_provider_opts if self.use_cuda else cpu_provider_opts
            )
            EP_list.insert(0, (EP.DIRECTML_EP.value, directml_options))
        return EP_list

    def _check_cuda(self) -> bool:
        if not self.cfg_use_cuda:
            return False

        cur_device = get_device()
        if cur_device == "GPU" and EP.CUDA_EP.value in self.had_providers:
            return True

        self.logger.warning(
            "%s is not in available providers (%s). Use %s inference by default.",
            EP.CUDA_EP.value,
            self.had_providers,
            self.had_providers[0],
        )
        self.logger.info("!!!Recommend to use rapidocr_paddle for inference on GPU.")
        self.logger.info(
            "(For reference only) If you want to use GPU acceleration, you must do:"
        )
        self.logger.info(
            "First, uninstall all onnxruntime pakcages in current environment."
        )
        self.logger.info(
            "Second, install onnxruntime-gpu by `pip install onnxruntime-gpu`."
        )
        self.logger.info(
            "\tNote the onnxruntime-gpu version must match your cuda and cudnn version."
        )
        self.logger.info(
            "\tYou can refer this link: https://onnxruntime.ai/docs/execution-providers/CUDA-EP.html"
        )
        self.logger.info(
            "Third, ensure %s is in available providers list. e.g. ['CUDAExecutionProvider', 'CPUExecutionProvider']",
            EP.CUDA_EP.value,
        )
        return False

    def _check_dml(self) -> bool:
        if not self.cfg_use_dml:
            return False

        cur_os = platform.system()
        if cur_os != "Windows":
            self.logger.warning(
                "DirectML is only supported in Windows OS. The current OS is %s. Use %s inference by default.",
                cur_os,
                self.had_providers[0],
            )
            return False

        cur_window_version = int(platform.release().split(".")[0])
        if cur_window_version < 10:
            self.logger.warning(
                "DirectML is only supported in Windows 10 and above OS. The current Windows version is %s. Use %s inference by default.",
                cur_window_version,
                self.had_providers[0],
            )
            return False

        if EP.DIRECTML_EP.value in self.had_providers:
            return True

        self.logger.warning(
            "%s is not in available providers (%s). Use %s inference by default.",
            EP.DIRECTML_EP.value,
            self.had_providers,
            self.had_providers[0],
        )
        self.logger.info("If you want to use DirectML acceleration, you must do:")
        self.logger.info(
            "First, uninstall all onnxruntime pakcages in current environment."
        )
        self.logger.info(
            "Second, install onnxruntime-directml by `pip install onnxruntime-directml`"
        )
        self.logger.info(
            "Third, ensure %s is in available providers list. e.g. ['DmlExecutionProvider', 'CPUExecutionProvider']",
            EP.DIRECTML_EP.value,
        )
        return False

    def _verify_providers(self):
        session_providers = self.session.get_providers()
        first_provider = session_providers[0]

        if self.use_cuda and first_provider != EP.CUDA_EP.value:
            self.logger.warning(
                "%s is not avaiable for current env, the inference part is automatically shifted to be executed under %s.",
                EP.CUDA_EP.value,
                first_provider,
            )

        if self.use_directml and first_provider != EP.DIRECTML_EP.value:
            self.logger.warning(
                "%s is not available for current env, the inference part is automatically shifted to be executed under %s.",
                EP.DIRECTML_EP.value,
                first_provider,
            )

    def __call__(self, input_content: List[np.ndarray]) -> np.ndarray:
        input_dict = dict(zip(self.get_input_names(), input_content))
        try:
            return self.session.run(None, input_dict)
        except Exception as e:
            error_info = traceback.format_exc()
            raise ONNXRuntimeError(error_info) from e

    def get_input_names(self) -> List[str]:
        return [v.name for v in self.session.get_inputs()]

    def get_output_names(self) -> List[str]:
        return [v.name for v in self.session.get_outputs()]

    def get_character_list(self, key: str = "character") -> List[str]:
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        return meta_dict[key].splitlines()

    def have_key(self, key: str = "character") -> bool:
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        if key in meta_dict.keys():
            return True
        return False

    @staticmethod
    def _verify_model(model_path: Union[str, Path, None]):
        if model_path is None:
            raise ValueError("model_path is None!")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exists.")

        if not model_path.is_file():
            raise FileExistsError(f"{model_path} is not a file.")


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

        origin_img_type = type(img)
        img = self.load_img(img)
        img = self.convert_img(img, origin_img_type)
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

        if isinstance(img, Image.Image):
            return np.array(img)

        raise LoadImageError(f"{type(img)} is not supported!")

    def convert_img(self, img: np.ndarray, origin_img_type):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.ndim == 3:
            channel = img.shape[2]
            if channel == 1:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            if channel == 2:
                return self.cvt_two_to_three(img)

            if channel == 3:
                if issubclass(origin_img_type, (str, Path, bytes, Image.Image)):
                    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return img

            if channel == 4:
                return self.cvt_four_to_three(img)

            raise LoadImageError(
                f"The channel({channel}) of the img is not in [1, 2, 3, 4]"
            )

        raise LoadImageError(f"The ndim({img.ndim}) of the img is not in [2, 3]")

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
    def verify_exist(file_path: Union[str, Path]):
        if not Path(file_path).exists():
            raise LoadImageError(f"{file_path} does not exist.")


class LoadImageError(Exception):
    pass


class VisTable:
    def __init__(self):
        self.load_img = LoadImage()

    def __call__(
        self,
        img_path: Union[str, Path],
        table_results,
        save_html_path: Optional[Union[str, Path]] = None,
        save_drawed_path: Optional[Union[str, Path]] = None,
        save_logic_path: Optional[Union[str, Path]] = None,
    ):
        if save_html_path:
            html_with_border = self.insert_border_style(table_results.pred_html)
            self.save_html(save_html_path, html_with_border)

        table_cell_bboxes = table_results.cell_bboxes
        table_logic_points = table_results.logic_points
        if table_cell_bboxes is None:
            return None

        img = self.load_img(img_path)

        dims_bboxes = table_cell_bboxes.shape[1]
        if dims_bboxes == 4:
            drawed_img = self.draw_rectangle(img, table_cell_bboxes)
        elif dims_bboxes == 8:
            drawed_img = self.draw_polylines(img, table_cell_bboxes)
        else:
            raise ValueError("Shape of table bounding boxes is not between in 4 or 8.")

        if save_drawed_path:
            self.save_img(save_drawed_path, drawed_img)

        if save_logic_path:
            polygons = [[box[0], box[1], box[4], box[5]] for box in table_cell_bboxes]
            self.plot_rec_box_with_logic_info(
                img_path, save_logic_path, table_logic_points, polygons
            )
        return drawed_img

    def insert_border_style(self, table_html_str: str):
        style_res = """<meta charset="UTF-8"><style>
        table {
            border-collapse: collapse;
            width: 100%;
        }
        th, td {
            border: 1px solid black;
            padding: 8px;
            text-align: center;
        }
        th {
            background-color: #f2f2f2;
        }
                    </style>"""

        prefix_table, suffix_table = table_html_str.split("<body>")
        html_with_border = f"{prefix_table}{style_res}<body>{suffix_table}"
        return html_with_border

    def plot_rec_box_with_logic_info(
        self, img_path, output_path, logic_points, sorted_polygons
    ):
        """
        :param img_path
        :param output_path
        :param logic_points: [row_start,row_end,col_start,col_end]
        :param sorted_polygons: [xmin,ymin,xmax,ymax]
        :return:
        """
        # 读取原图
        img = cv2.imread(img_path)
        img = cv2.copyMakeBorder(
            img, 0, 0, 0, 100, cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
        # 绘制 polygons 矩形
        for idx, polygon in enumerate(sorted_polygons):
            x0, y0, x1, y1 = polygon[0], polygon[1], polygon[2], polygon[3]
            x0 = round(x0)
            y0 = round(y0)
            x1 = round(x1)
            y1 = round(y1)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
            # 增大字体大小和线宽
            font_scale = 0.9  # 原先是0.5
            thickness = 1  # 原先是1
            logic_point = logic_points[idx]
            cv2.putText(
                img,
                f"row: {logic_point[0]}-{logic_point[1]}",
                (x0 + 3, y0 + 8),
                cv2.FONT_HERSHEY_PLAIN,
                font_scale,
                (0, 0, 255),
                thickness,
            )
            cv2.putText(
                img,
                f"col: {logic_point[2]}-{logic_point[3]}",
                (x0 + 3, y0 + 18),
                cv2.FONT_HERSHEY_PLAIN,
                font_scale,
                (0, 0, 255),
                thickness,
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # 保存绘制后的图像
            self.save_img(output_path, img)

    @staticmethod
    def draw_rectangle(img: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        img_copy = img.copy()
        for box in boxes.astype(int):
            x1, y1, x2, y2 = box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return img_copy

    @staticmethod
    def draw_polylines(img: np.ndarray, points) -> np.ndarray:
        img_copy = img.copy()
        for point in points.astype(int):
            point = point.reshape(4, 2)
            cv2.polylines(img_copy, [point.astype(int)], True, (255, 0, 0), 2)
        return img_copy

    @staticmethod
    def save_img(save_path: Union[str, Path], img: np.ndarray):
        cv2.imwrite(str(save_path), img)

    @staticmethod
    def save_html(save_path: Union[str, Path], html: str):
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html)
