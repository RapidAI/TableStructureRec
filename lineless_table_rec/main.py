# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import argparse
import logging
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

from .lineless_table_process import DetProcess, get_affine_transform_upper_left
from .utils import LoadImage, OrtInferSession
from .utils_table_recover import (
    get_rotate_crop_image,
    match_ocr_cell,
    plot_html_table,
    sorted_boxes,
)

cur_dir = Path(__file__).resolve().parent
detect_model_path = cur_dir / "models" / "lore_detect.onnx"
process_model_path = cur_dir / "models" / "lore_process.onnx"


class LinelessTableRecognition:
    def __init__(
        self,
    ):
        self.mean = np.array([0.408, 0.447, 0.470], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.289, 0.274, 0.278], dtype=np.float32).reshape(1, 1, 3)

        self.inp_h = 768
        self.inp_w = 768

        self.det_session = OrtInferSession(detect_model_path)
        self.process_session = OrtInferSession(process_model_path)

        self.load_img = LoadImage()
        self.det_process = DetProcess()
        self.ocr = RapidOCR()

    def __call__(self, content: Dict[str, Any]) -> str:
        ss = time.perf_counter()
        img = self.load_img(content)

        ocr_res, _ = self.ocr(img)

        input_info = self.preprocess(img)
        try:
            polygons, slct_logi = self.infer(input_info)
            logi_points = self.filter_logi_points(slct_logi)

            sorted_polygons = sorted_boxes(polygons)

            cell_box_map = match_ocr_cell(sorted_polygons, ocr_res)
            cell_box_map = self.re_rec(img, sorted_polygons, cell_box_map)

            logi_points = self.sort_logi_by_polygons(
                sorted_polygons, polygons, logi_points
            )

            table_str = plot_html_table(logi_points, cell_box_map)
            table_elapse = time.perf_counter() - ss
            return table_str, table_elapse
        except Exception:
            logging.warning(traceback.format_exc())
            return "", 0.0

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

    def infer(self, input: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        hm, st, wh, ax, cr, reg = self.det_session([input["img"]])
        output = {
            "hm": hm,
            "st": st,
            "wh": wh,
            "ax": ax,
            "cr": cr,
            "reg": reg,
        }
        slct_logi_feat, slct_dets_feat, slct_output_dets = self.det_process(
            output, input["meta"]
        )

        slct_output_dets = slct_output_dets.reshape(-1, 4, 2)

        _, slct_logi = self.process_session(
            [slct_logi_feat, slct_dets_feat.astype(np.int64)]
        )
        return slct_output_dets, slct_logi

    def filter_logi_points(self, slct_logi: np.ndarray) -> Dict[str, Any]:
        logi_floor = np.floor(slct_logi)
        dev = slct_logi - logi_floor
        slct_logi = np.where(dev > 0.5, logi_floor + 1, logi_floor)
        return slct_logi[0]

    @staticmethod
    def sort_logi_by_polygons(
        sorted_polygons: np.ndarray, polygons: np.ndarray, logi_points: np.ndarray
    ) -> np.ndarray:
        sorted_idx = []
        for v in sorted_polygons:
            loc_idx = np.argwhere(v[0, 0] == polygons[:, 0, 0]).squeeze()
            sorted_idx.append(int(loc_idx))
        logi_points = logi_points[sorted_idx]
        return logi_points

    def re_rec(
        self,
        img: np.ndarray,
        sorted_polygons: np.ndarray,
        cell_box_map: Dict[int, List[str]],
    ) -> Dict[int, List[str]]:
        """找到poly对应为空的框，尝试将直接将poly框直接送到识别中"""
        for k, v in cell_box_map.items():
            if v[0]:
                continue

            crop_img = get_rotate_crop_image(img, sorted_polygons[k])
            pad_img = cv2.copyMakeBorder(
                crop_img, 2, 2, 100, 100, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            rec_res, _ = self.ocr(pad_img, use_det=False, use_cls=True, use_rec=True)
            cell_box_map[k] = [rec_res[0][0]]
        return cell_box_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-img", "--img_path", type=str, required=True)
    args = parser.parse_args()

    table_rec = LinelessTableRecognition()
    table_str, elapse = table_rec(args.img_path)
    print(table_str)
    print(f"cost: {elapse:.5f}")


if __name__ == "__main__":
    main()
