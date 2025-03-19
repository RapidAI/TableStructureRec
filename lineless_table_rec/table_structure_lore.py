from copy import deepcopy
from typing import Dict, Any, Tuple, Optional

import cv2
import numpy as np

from .utils.utils import OrtInferSession
from .utils.utils_table_lore_rec import DetProcess, get_affine_transform_upper_left


class TSRLore:
    def __init__(self, config: Dict):
        self.mean = np.array([0.408, 0.447, 0.470], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.289, 0.274, 0.278], dtype=np.float32).reshape(1, 1, 3)

        self.inp_h = 768
        self.inp_w = 768

        det_config = deepcopy(config)
        process_config = deepcopy(config)
        det_config["model_path"] = config["model_path"]["lore_detect"]
        process_config["model_path"] = config["model_path"]["lore_process"]
        self.det_session = OrtInferSession(det_config)
        self.process_session = OrtInferSession(process_config)
        self.det_process = DetProcess()

    def __call__(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        img_info = self.preprocess(img)
        polygons, slct_logi = self.infer(img_info)
        logi_points = self.postprocess(slct_logi)
        return polygons, logi_points

    def preprocess(self, img: np.ndarray) -> Dict[str, Any]:
        height, width = img.shape[:2]
        resized_image = cv2.resize(img, (width, height))

        c = np.array([0, 0], dtype=np.float32)
        s = max(height, width) * 1.0
        trans_input = get_affine_transform_upper_left(c, s, [self.inp_w, self.inp_h])

        inp_image = cv2.warpAffine(
            resized_image, trans_input, (self.inp_w, self.inp_h), flags=cv2.INTER_LINEAR
        )
        inp_image = ((inp_image / 255.0 - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, self.inp_h, self.inp_w)
        meta = {
            "c": c,
            "s": s,
            "out_height": self.inp_h // 4,
            "out_width": self.inp_w // 4,
        }
        return {"img": images, "meta": meta}

    def infer(self, input_content: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        hm, st, wh, ax, cr, reg = self.det_session([input_content["img"]])
        output = {
            "hm": hm,
            "st": st,
            "wh": wh,
            "ax": ax,
            "cr": cr,
            "reg": reg,
        }
        slct_logi_feat, slct_dets_feat, slct_output_dets = self.det_process(
            output, input_content["meta"]
        )

        slct_output_dets = slct_output_dets.reshape(-1, 4, 2)

        _, slct_logi = self.process_session(
            [slct_logi_feat, slct_dets_feat.astype(np.int64)]
        )
        return slct_output_dets, slct_logi

    def postprocess(self, slct_logi: np.ndarray) -> np.ndarray:
        for logic_points in slct_logi[0]:
            # 修正坐标接近导致的r_e > r_s 或 c_e > c_s
            if abs(logic_points[0] - logic_points[1]) < 0.2:
                row = (logic_points[0] + logic_points[1]) / 2
                logic_points[0] = row
                logic_points[1] = row
            if abs(logic_points[2] - logic_points[3]) < 0.2:
                col = (logic_points[2] + logic_points[3]) / 2
                logic_points[2] = col
                logic_points[3] = col
        logi_floor = np.floor(slct_logi)
        dev = slct_logi - logi_floor
        slct_logi = np.where(dev > 0.5, logi_floor + 1, logi_floor)
        return slct_logi[0].astype(np.int32)
