# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import argparse
import importlib
import logging
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np
import cv2

from wired_table_rec.table_line_rec import TableLineRecognition
from wired_table_rec.table_line_rec_plus import TableLineRecognitionPlus
from .table_recover import TableRecover
from .utils import InputType, LoadImage
from .utils_table_recover import (
    match_ocr_cell,
    plot_html_table,
    box_4_2_poly_to_box_4_1,
    get_rotate_crop_image,
    sorted_ocr_boxes,
    gather_ocr_list_by_row,
)

cur_dir = Path(__file__).resolve().parent
default_model_path = cur_dir / "models" / "cycle_center_net_v1.onnx"
default_model_path_v2 = cur_dir / "models" / "cycle_center_net_v2.onnx"


class WiredTableRecognition:
    def __init__(self, table_model_path: Union[str, Path] = None, version="v2"):
        self.load_img = LoadImage()
        if version == "v2":
            model_path = table_model_path if table_model_path else default_model_path_v2
            self.table_line_rec = TableLineRecognitionPlus(str(model_path))
        else:
            model_path = table_model_path if table_model_path else default_model_path
            self.table_line_rec = TableLineRecognition(str(model_path))

        self.table_recover = TableRecover()

        try:
            self.ocr = importlib.import_module("rapidocr_onnxruntime").RapidOCR()
        except ModuleNotFoundError:
            self.ocr = None

    def __call__(
        self,
        img: InputType,
        ocr_result: Optional[List[Union[List[List[float]], str, str]]] = None,
        **kwargs,
    ) -> Tuple[str, float, Any, Any, Any]:
        if self.ocr is None and ocr_result is None:
            raise ValueError(
                "One of two conditions must be met: ocr_result is not empty, or rapidocr_onnxruntime is installed."
            )

        s = time.perf_counter()
        rec_again = True
        need_ocr = True
        col_threshold = 15
        row_threshold = 10
        if kwargs:
            rec_again = kwargs.get("rec_again", True)
            need_ocr = kwargs.get("need_ocr", True)
            col_threshold = kwargs.get("col_threshold", 15)
            row_threshold = kwargs.get("row_threshold", 10)
        img = self.load_img(img)
        polygons, rotated_polygons = self.table_line_rec(img, **kwargs)
        if polygons is None:
            logging.warning("polygons is None.")
            return "", 0.0, None, None, None

        try:
            table_res, logi_points = self.table_recover(
                rotated_polygons, row_threshold, col_threshold
            )
            # 将坐标由逆时针转为顺时针方向，后续处理与无线表格对齐
            polygons[:, 1, :], polygons[:, 3, :] = (
                polygons[:, 3, :].copy(),
                polygons[:, 1, :].copy(),
            )
            if not need_ocr:
                sorted_polygons, idx_list = sorted_ocr_boxes(
                    [box_4_2_poly_to_box_4_1(box) for box in polygons]
                )
                return (
                    "",
                    time.perf_counter() - s,
                    sorted_polygons,
                    logi_points[idx_list],
                    [],
                )
            if ocr_result is None and need_ocr:
                ocr_result, _ = self.ocr(img)
            cell_box_det_map, not_match_orc_boxes = match_ocr_cell(ocr_result, polygons)
            # 如果有识别框没有ocr结果，直接进行rec补充
            cell_box_det_map = self.re_rec(img, polygons, cell_box_det_map, rec_again)
            # 转换为中间格式，修正识别框坐标,将物理识别框，逻辑识别框，ocr识别框整合为dict，方便后续处理
            t_rec_ocr_list_dict = self.transform_res(cell_box_det_map, polygons, logi_points)
            # 第一行或者第一列为空时，调整代码
            #adjust_dict = self.adjust_table_cells(t_rec_ocr_list_dict)
            adjust_dict = self.process_ocr_result(t_rec_ocr_list_dict)
            # 将每个单元格中的ocr识别结果排序和同行合并，输出的html能完整保留文字的换行格式
            t_rec_ocr_list = self.sort_and_gather_ocr_res(t_rec_ocr_list_dict)
            # cell_box_map =
            logi_points = [t_box_ocr["t_logic_box"] for t_box_ocr in t_rec_ocr_list]
            cell_box_det_map = {
                i: [ocr_box_and_text[1] for ocr_box_and_text in t_box_ocr["t_ocr_res"]]
                for i, t_box_ocr in enumerate(t_rec_ocr_list)
            }
            table_str = plot_html_table(logi_points, cell_box_det_map)
            ocr_boxes_res = [
                box_4_2_poly_to_box_4_1(ori_ocr[0]) for ori_ocr in ocr_result
            ]
            sorted_ocr_boxes_res, _ = sorted_ocr_boxes(ocr_boxes_res)
            sorted_polygons = [box_4_2_poly_to_box_4_1(box) for box in polygons]
            sorted_logi_points = logi_points
            table_elapse = time.perf_counter() - s

        except Exception:
            logging.warning(traceback.format_exc())
            return "", 0.0, None, None, None
        return (
            table_str,
            table_elapse,
            sorted_polygons,
            sorted_logi_points,
            sorted_ocr_boxes_res,
            adjust_dict

        )

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

    def process_ocr_result(self, ocr_result):
        # 删除第一行的字典，并调整其余字典的行数
        first_row_empty = [entry for entry in ocr_result if
                           entry['t_logic_box'][0] == 0 and entry['t_logic_box'][1] == 0 and entry['t_ocr_res'][0][
                               1] == '']

        if len(first_row_empty) == len(
                [entry for entry in ocr_result if entry['t_logic_box'][0] == 0 and entry['t_logic_box'][1] == 0]):
            # 如果第一行的所有单元格都为空，删除第一行
            ocr_result = [entry for entry in ocr_result if entry['t_logic_box'][0] != 0 or entry['t_logic_box'][1] != 0]
            # 调整剩余字典的行数
            for entry in ocr_result:
                entry['t_logic_box'][0] -= 1
                entry['t_logic_box'][1] -= 1

        # 删除第一列的字典，并调整其余字典的列数
        first_col_empty = [entry for entry in ocr_result if
                           entry['t_logic_box'][2] == 0 and entry['t_logic_box'][3] == 0 and entry['t_ocr_res'][0][
                               1] == '']

        if len(first_col_empty) == len(
                [entry for entry in ocr_result if entry['t_logic_box'][2] == 0 and entry['t_logic_box'][3] == 0]):
            # 如果第一列的所有单元格都为空，删除第一列
            ocr_result = [entry for entry in ocr_result if entry['t_logic_box'][2] != 0 or entry['t_logic_box'][3] != 0]
            # 调整剩余字典的列数
            for entry in ocr_result:
                entry['t_logic_box'][2] -= 1
                entry['t_logic_box'][3] -= 1

        return ocr_result

    def adjust_table_cells(self, t_rec_ocr_list_dict):
        """
        调整表格单元格，去掉第一行和/或第一列的单元格，
        并更新剩余单元格的行列起始和结束位置。

        参数:
            t_rec_ocr_list_dict (list): 原始表格单元格识别结果，格式为
                [
                    {
                        "t_box": [xmin, ymin, xmax, ymax],
                        "t_logic_box": [row_start, row_end, col_start, col_end],
                        "t_ocr_res": [[box, text], ...]
                    },
                    ...
                ]

        返回:
            list: 调整后的表格单元格识别结果，格式与输入相同。
        """
        # 新的结果列表
        adjusted_result = []

        # 记录是否第一行和第一列的单元格已被删除
        remove_first_row = False
        remove_first_col = False

        # 检查并移除第一行
        if all(cell and not cell[1] for cell in t_rec_ocr_list_dict[0].get("t_ocr_res", [])):
            remove_first_row = True

        # 检查并移除第一列
        if all(row.get("t_ocr_res") and not row["t_ocr_res"][0][1] for row in t_rec_ocr_list_dict):
            remove_first_col = True

        # 遍历原始结果进行调整
        for i, row in enumerate(t_rec_ocr_list_dict):
            adjusted_row = []

            # 如果是第一行并且需要删除，跳过这行
            if remove_first_row and i == 0:
                continue

            for j, cell in enumerate(row.get("t_ocr_res", [])):
                # 如果是第一列并且需要删除，跳过这一列
                if remove_first_col and j == 0:
                    continue

                # 更新当前单元格的逻辑位置
                adjusted_cell = {
                    "t_box": row.get("t_box"),
                    "t_logic_box": [
                        row["t_logic_box"][0] - 1 if i > 0 else row["t_logic_box"][0],
                        row["t_logic_box"][1] - 1 if i > 0 else row["t_logic_box"][1],
                        row["t_logic_box"][2] - 1 if j > 0 else row["t_logic_box"][2],
                        row["t_logic_box"][3] - 1 if j > 0 else row["t_logic_box"][3]
                    ],
                    "t_ocr_res": cell
                }
                adjusted_row.append(adjusted_cell)

            if adjusted_row:
                adjusted_result.append(adjusted_row)

        return adjusted_result

    def sort_and_gather_ocr_res(self, res):
        for i, dict_res in enumerate(res):
            _, sorted_idx = sorted_ocr_boxes(
                [ocr_det[0] for ocr_det in dict_res["t_ocr_res"]], threhold=0.3
            )
            dict_res["t_ocr_res"] = [dict_res["t_ocr_res"][i] for i in sorted_idx]
            dict_res["t_ocr_res"] = gather_ocr_list_by_row(
                dict_res["t_ocr_res"], threhold=0.3
            )
        return res

    def re_rec(
        self,
        img: np.ndarray,
        sorted_polygons: np.ndarray,
        cell_box_map: Dict[int, List[str]],
        rec_again=True,
    ) -> Dict[int, List[Any]]:
        """找到poly对应为空的框，尝试将直接将poly框直接送到识别中"""
        for i in range(sorted_polygons.shape[0]):
            if cell_box_map.get(i):
                continue
            if not rec_again:
                box = sorted_polygons[i]
                cell_box_map[i] = [[box, "", 1]]
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

    def re_rec_high_precise(
        self,
        img: np.ndarray,
        sorted_polygons: np.ndarray,
        cell_box_map: Dict[int, List[str]],
    ) -> Dict[int, List[any]]:
        """找到poly对应为空的框，尝试将直接将poly框直接送到识别中"""
        #
        cell_box_map = {}
        for i in range(sorted_polygons.shape[0]):
            if cell_box_map.get(i):
                continue
            crop_img = get_rotate_crop_image(img, sorted_polygons[i])
            pad_img = cv2.copyMakeBorder(
                crop_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            rec_res, _ = self.ocr(pad_img, use_det=True, use_cls=True, use_rec=True)
            if not rec_res:
                det_boxes = [sorted_polygons[i]]
                text = [""]
                scores = [1.0]
            else:
                det_boxes = [rec[0] for rec in rec_res]
                text = [rec[1] for rec in rec_res]
                scores = [rec[2] for rec in rec_res]
            cell_box_map[i] = [
                [box, text, score] for box, text, score in zip(det_boxes, text, scores)
            ]
        return cell_box_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-img", "--img_path", type=str, required=True)
    args = parser.parse_args()

    try:
        ocr_engine = importlib.import_module("rapidocr_onnxruntime").RapidOCR()
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Please install the rapidocr_onnxruntime by pip install rapidocr_onnxruntime."
        ) from exc

    table_rec = WiredTableRecognition()
    ocr_result, _ = ocr_engine(args.img_path)
    table_str, elapse = table_rec(args.img_path, ocr_result)
    print(table_str)
    print(f"cost: {elapse:.5f}")


if __name__ == "__main__":
    main()
