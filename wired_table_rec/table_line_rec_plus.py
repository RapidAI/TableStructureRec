import copy
import math
from typing import Optional, Dict, Any

import cv2
import numpy as np
from skimage import measure

from wired_table_rec.utils import OrtInferSession, resize_img
from wired_table_rec.utils_table_line_rec import (
    get_table_line,
    final_adjust_lines,
    min_area_rect_box,
    draw_lines,
    adjust_lines,
)
from wired_table_rec.utils_table_recover import (
    sorted_ocr_boxes,
    box_4_2_poly_to_box_4_1,
)


class TableLineRecognitionPlus:
    def __init__(self, model_path: Optional[str] = None):
        self.K = 1000
        self.MK = 4000
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        self.inp_height = 1024
        self.inp_width = 1024

        self.session = OrtInferSession(model_path)

    def __call__(self, img: np.ndarray) -> Optional[np.ndarray]:
        img_info = self.preprocess(img)
        pred = self.infer(img_info)
        polygons = self.postprocess(img, pred)
        if polygons.size == 0:
            return None

        polygons = polygons.reshape(polygons.shape[0], 4, 2)
        polygons[:, 3, :], polygons[:, 1, :] = (
            polygons[:, 1, :].copy(),
            polygons[:, 3, :].copy(),
        )
        _, idx = sorted_ocr_boxes(
            [box_4_2_poly_to_box_4_1(poly_box) for poly_box in polygons], threhold=0.4
        )
        polygons = polygons[idx]
        return polygons

    def preprocess(self, img) -> Dict[str, Any]:
        scale = (self.inp_height, self.inp_width)
        img, _, _ = resize_img(img, scale, True)
        img = img.copy().astype(np.float32)
        assert img.dtype != np.uint8
        mean = np.float64(self.mean.reshape(1, -1))
        stdinv = 1 / np.float64(self.std.reshape(1, -1))
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        img = img.transpose(2, 0, 1)
        images = img[None, :]
        return {"img": images}

    def infer(self, input):
        result = self.session(input["img"][None, ...])[0][0]
        result = result[0].astype(np.uint8)
        return result

    def postprocess(self, img, pred, row=50, col=30, alph=15, angle=50):
        ori_shape = img.shape
        pred = np.uint8(pred)
        hpred = copy.deepcopy(pred)  # 横线
        vpred = copy.deepcopy(pred)  # 竖线
        whereh = np.where(hpred == 1)
        wherev = np.where(vpred == 2)
        hpred[wherev] = 0
        vpred[whereh] = 0

        hpred = cv2.resize(hpred, (ori_shape[1], ori_shape[0]))
        vpred = cv2.resize(vpred, (ori_shape[1], ori_shape[0]))

        h, w = pred.shape
        hors_k = int(math.sqrt(w) * 1.2)
        vert_k = int(math.sqrt(h) * 1.2)
        hkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hors_k, 1))
        vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_k))
        vpred = cv2.morphologyEx(
            vpred, cv2.MORPH_CLOSE, vkernel, iterations=1
        )  # 先膨胀后腐蚀的过程
        hpred = cv2.morphologyEx(hpred, cv2.MORPH_CLOSE, hkernel, iterations=1)
        colboxes = get_table_line(vpred, axis=1, lineW=col)  # 竖线
        rowboxes = get_table_line(hpred, axis=0, lineW=row)  # 横线
        # rboxes_row_, rboxes_col_ = adjust_lines(rowboxes, colboxes, alph = alph, angle=angle)
        rboxes_row_ = adjust_lines(rowboxes, alph=100, angle=angle)
        rboxes_col_ = adjust_lines(colboxes, alph=alph, angle=angle)
        rowboxes += rboxes_row_
        colboxes += rboxes_col_
        rowboxes, colboxes = final_adjust_lines(rowboxes, colboxes)

        tmp = np.zeros(img.shape[:2], dtype="uint8")
        tmp = draw_lines(tmp, rowboxes + colboxes, color=255, lineW=2)
        labels = measure.label(tmp < 255, connectivity=2)  # 8连通区域标记
        regions = measure.regionprops(labels)
        ceilboxes = min_area_rect_box(
            regions,
            False,
            tmp.shape[1],
            tmp.shape[0],
            filtersmall=True,
            adjust_box=False,
        )  # 最后一个参数改为False
        return np.array(ceilboxes)
