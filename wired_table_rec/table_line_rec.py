# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .utils import OrtInferSession
from .utils_table_line_rec import (
    bbox_decode,
    bbox_post_process,
    gbox_decode,
    gbox_post_process,
    get_affine_transform,
    group_bbox_by_gbox,
    nms,
)
from .utils_table_recover import merge_adjacent_polys, sorted_boxes


class TableLineRecognition:
    def __init__(self, model_path: str = None):
        self.K = 1000
        self.MK = 4000
        self.mean = np.array([0.408, 0.447, 0.470], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.289, 0.274, 0.278], dtype=np.float32).reshape(1, 1, 3)

        self.inp_height = 1024
        self.inp_width = 1024

        self.session = OrtInferSession(model_path)

    def __call__(self, img: np.ndarray) -> Optional[np.ndarray]:
        img_info = self.preprocess(img)
        pred = self.infer(img_info)
        polygons = self.postprocess(pred)
        if polygons.size == 0:
            return None

        polygons = polygons.reshape(polygons.shape[0], 4, 2)
        polygons = sorted_boxes(polygons)
        polygons = merge_adjacent_polys(polygons)
        return polygons

    def preprocess(self, img) -> Dict[str, Any]:
        height, width = img.shape[:2]

        resized_image = cv2.resize(img, (width, height))

        c = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        s = max(height, width) * 1.0
        trans_input = get_affine_transform(c, s, 0, [self.inp_width, self.inp_height])

        inp_image = cv2.warpAffine(
            resized_image,
            trans_input,
            (self.inp_width, self.inp_height),
            flags=cv2.INTER_LINEAR,
        )

        inp_image = ((inp_image / 255.0 - self.mean) / self.std).astype(np.float32)
        images = inp_image.transpose(2, 0, 1).reshape(
            1, 3, self.inp_height, self.inp_width
        )
        meta = {
            "c": c,
            "s": s,
            "input_height": self.inp_height,
            "input_width": self.inp_width,
            "out_height": self.inp_height // 4,
            "out_width": self.inp_width // 4,
        }
        return {"img": images, "meta": meta}

    def infer(self, input):
        ort_outs = self.session(input["img"][None, ...])
        pred = [
            {
                "hm": ort_outs[0],
                "v2c": ort_outs[1],
                "c2v": ort_outs[2],
                "reg": ort_outs[3],
            }
        ]
        return {"results": pred, "meta": input["meta"]}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        output = inputs["results"][0]
        meta = inputs["meta"]

        hm = self.sigmoid(output["hm"])
        v2c = output["v2c"]
        c2v = output["c2v"]
        reg = output["reg"]

        bbox, _ = bbox_decode(hm[:, 0:1, :, :], c2v, reg=reg, K=self.K)
        gbox, _ = gbox_decode(hm[:, 1:2, :, :], v2c, reg=reg, K=self.MK)

        bbox = nms(bbox, 0.3)
        c, s, h, w = [meta["c"]], [meta["s"]], meta["out_height"], meta["out_width"]
        bbox = bbox_post_process(bbox.copy(), c, s, h, w)
        gbox = gbox_post_process(gbox.copy(), c, s, h, w)

        bbox = group_bbox_by_gbox(bbox[0], gbox[0])
        polygons = [box[:8] for box in bbox if box[8] > 0.3]
        return np.array(polygons)

    @staticmethod
    def sigmoid(data: np.ndarray):
        return 1 / (1 + np.exp(-data))
