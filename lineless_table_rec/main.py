# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import logging
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

from .process import DetProcess, get_affine_transform_upper_left
from .utils import InputType, LoadImage, OrtInferSession
from .utils_table_recover import (
    box_4_2_poly_to_box_4_1,
    filter_duplicated_box,
    gather_ocr_list_by_row,
    get_rotate_crop_image,
    match_ocr_cell,
    plot_html_table,
    sorted_ocr_boxes,
)

cur_dir = Path(__file__).resolve().parent
detect_model_path = cur_dir / "models" / "lore_detect.onnx"
process_model_path = cur_dir / "models" / "lore_process.onnx"


class LinelessTableRecognition:
    def __init__(
        self,
        detect_model_path: Union[str, Path] = detect_model_path,
        process_model_path: Union[str, Path] = process_model_path,
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

    def __call__(
        self,
        content: InputType,
        ocr_result: Optional[List[Union[List[List[float]], str, str]]] = None,
    ):
        ss = time.perf_counter()
        img = self.load_img(content)
        if self.ocr is None and ocr_result is None:
            raise ValueError(
                "One of two conditions must be met: ocr_result is not empty, or rapidocr_onnxruntime is installed."
            )
        if ocr_result is None:
            ocr_result, _ = self.ocr(img)
        input_info = self.preprocess(img)
        try:
            polygons, slct_logi = self.infer(input_info)
            logi_points = self.filter_logi_points(slct_logi)
            # ocr 结果匹配
            cell_box_det_map, no_match_ocr_det = match_ocr_cell(ocr_result, polygons)
            # 如果有识别框没有ocr结果，直接进行rec补充
            cell_box_det_map = self.re_rec(img, polygons, cell_box_det_map)
            # 转换为中间格式，修正识别框坐标,将物理识别框，逻辑识别框，ocr识别框整合为dict，方便后续处理
            t_rec_ocr_list = self.transform_res(cell_box_det_map, polygons, logi_points)
            # 拆分包含和重叠的识别框
            deleted_idx_set = filter_duplicated_box(
                [table_box_ocr["t_box"] for table_box_ocr in t_rec_ocr_list]
            )
            t_rec_ocr_list = [
                t_rec_ocr_list[i]
                for i in range(len(t_rec_ocr_list))
                if i not in deleted_idx_set
            ]
            # 生成行列对应的二维表格, 合并同行同列识别框中的的ocr识别框
            t_rec_ocr_list, grid = self.handle_overlap_row_col(t_rec_ocr_list)
            # todo 根据grid 及 not_match_orc_boxes，尝试将ocr识别填入单行单列中
            # 将同一个识别框中的ocr结果排序并同行合并
            t_rec_ocr_list = self.sort_and_gather_ocr_res(t_rec_ocr_list)
            # 渲染为html
            logi_points = [t_box_ocr["t_logic_box"] for t_box_ocr in t_rec_ocr_list]
            cell_box_det_map = {
                i: [ocr_box_and_text[1] for ocr_box_and_text in t_box_ocr["t_ocr_res"]]
                for i, t_box_ocr in enumerate(t_rec_ocr_list)
            }
            table_str = plot_html_table(logi_points, cell_box_det_map)

            # 输出可视化排序,用于验证结果，生产版本可以去掉
            _, idx_list = sorted_ocr_boxes(
                [t_box_ocr["t_box"] for t_box_ocr in t_rec_ocr_list]
            )
            t_rec_ocr_list = [t_rec_ocr_list[i] for i in idx_list]
            sorted_polygons = [t_box_ocr["t_box"] for t_box_ocr in t_rec_ocr_list]
            sorted_logi_points = [
                t_box_ocr["t_logic_box"] for t_box_ocr in t_rec_ocr_list
            ]
            ocr_boxes_res = [
                box_4_2_poly_to_box_4_1(ori_ocr[0]) for ori_ocr in ocr_result
            ]
            sorted_ocr_boxes_res, _ = sorted_ocr_boxes(ocr_boxes_res)
            table_elapse = time.perf_counter() - ss
            return (
                table_str,
                table_elapse,
                sorted_polygons,
                sorted_logi_points,
                sorted_ocr_boxes_res,
            )
        except Exception:
            logging.warning(traceback.format_exc())
            return "", 0.0, None, None, None

    def transform_res(
        self,
        cell_box_det_map: Dict[int, List[any]],
        polygons: np.ndarray,
        logi_points: List[np.ndarray],
    ) -> List[Dict[str, any]]:
        res = []
        for i in range(len(polygons)):
            ocr_res_list = cell_box_det_map.get(i)
            if not ocr_res_list:
                continue
            xmin = min([ocr_box[0][0][0] for ocr_box in ocr_res_list])
            ymin = min([ocr_box[0][0][1] for ocr_box in ocr_res_list])
            xmax = max([ocr_box[0][2][0] for ocr_box in ocr_res_list])
            ymax = max([ocr_box[0][2][1] for ocr_box in ocr_res_list])
            dict_res = {
                # xmin,xmax,ymin,ymax
                "t_box": [xmin, ymin, xmax, ymax],
                # row_start,row_end,col_start,col_end
                "t_logic_box": logi_points[i].tolist(),
                # [[xmin,xmax,ymin,ymax], text]
                "t_ocr_res": [
                    [box_4_2_poly_to_box_4_1(ocr_det[0]), ocr_det[1]]
                    for ocr_det in ocr_res_list
                ],
            }
            res.append(dict_res)
        return res

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

    def sort_and_gather_ocr_res(self, res):
        for i, dict_res in enumerate(res):
            _, sorted_idx = sorted_ocr_boxes(
                [ocr_det[0] for ocr_det in dict_res["t_ocr_res"]], threhold=0.5
            )
            dict_res["t_ocr_res"] = [dict_res["t_ocr_res"][i] for i in sorted_idx]
            dict_res["t_ocr_res"] = gather_ocr_list_by_row(
                dict_res["t_ocr_res"], thehold=0.5
            )
        return res

    def handle_overlap_row_col(self, res):
        max_row, max_col = 0, 0
        for dict_res in res:
            max_row = max(max_row, dict_res["t_logic_box"][1] + 1)  # 加1是因为结束下标是包含在内的
            max_col = max(max_col, dict_res["t_logic_box"][3] + 1)  # 加1是因为结束下标是包含在内的

        # 创建一个二维数组来存储 sorted_logi_points 中的元素
        grid = [[None] * max_col for _ in range(max_row)]

        # 将 sorted_logi_points 中的元素填充到 grid 中
        deleted_idx = set()
        for i, dict_res in enumerate(res):
            if i in deleted_idx:
                continue
            row_start, row_end, col_start, col_end = dict_res["t_logic_box"]
            for row in range(row_start, row_end + 1):
                if i in deleted_idx:
                    continue
                for col in range(col_start, col_end + 1):
                    if i in deleted_idx:
                        continue
                    exist_dict_res = grid[row][col]
                    if not exist_dict_res:
                        grid[row][col] = dict_res
                        continue
                    if exist_dict_res["t_logic_box"] == dict_res["t_logic_box"]:
                        exist_dict_res["t_ocr_res"].extend(dict_res["t_ocr_res"])
                        deleted_idx.add(i)
                        # 修正识别框坐标
                        exist_dict_res["t_box"] = [
                            min(exist_dict_res["t_box"][0], dict_res["t_box"][0]),
                            min(exist_dict_res["t_box"][1], dict_res["t_box"][1]),
                            max(exist_dict_res["t_box"][2], dict_res["t_box"][2]),
                            max(exist_dict_res["t_box"][3], dict_res["t_box"][3]),
                        ]
                        continue

        #  去掉重叠框
        res = [res[i] for i in range(len(res)) if i not in deleted_idx]
        return res, grid

    @staticmethod
    def filter_logi_points(slct_logi: np.ndarray) -> List[np.ndarray]:
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

    def re_rec(
        self,
        img: np.ndarray,
        sorted_polygons: np.ndarray,
        cell_box_map: Dict[int, List[str]],
    ) -> Dict[int, List[any]]:
        """找到poly对应为空的框，尝试将直接将poly框直接送到识别中"""
        #
        for i in range(sorted_polygons.shape[0]):
            if cell_box_map.get(i):
                continue
            crop_img = get_rotate_crop_image(img, sorted_polygons[i])
            pad_img = cv2.copyMakeBorder(
                crop_img, 5, 5, 100, 100, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            rec_res, _ = self.ocr(pad_img, use_det=False, use_cls=True, use_rec=True)
            box = sorted_polygons[i]
            text = [rec[0] for rec in rec_res]
            scores = [rec[1] for rec in rec_res]
            cell_box_map[i] = [[box, "".join(text), min(scores)]]
        return cell_box_map
