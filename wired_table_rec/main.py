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

from wired_table_rec.table_structure.utils import (
    box_4_2_poly_to_box_4_1,
    sorted_ocr_boxes,
    is_single_axis_contained,
    is_box_contained,
    calculate_iou,
)
from .table_structure.lore_wired_rec import LoreWiredRecognition
from .table_structure.unet_line_rec import UnetLineRecognition
from .table_match.table_recover import TableRecover
from .utils.utils import InputType, LoadImage


cur_dir = Path(__file__).resolve().parent
default_model_path = cur_dir / "models" / "cycle_center_net_v1.onnx"
default_model_path_v2 = cur_dir / "models" / "cycle_center_net_v2.onnx"


class WiredTableRecognition:
    def __init__(self, table_model_path: Union[str, Path] = None, version="v2"):
        self.load_img = LoadImage()
        if version == "v2":
            model_path = table_model_path if table_model_path else default_model_path_v2
            self.table_line_rec = UnetLineRecognition(str(model_path))
        else:
            model_path = table_model_path if table_model_path else default_model_path
            self.table_line_rec = LoreWiredRecognition(str(model_path))

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
            cell_box_det_map, not_match_orc_boxes = self.match_ocr_cell(
                ocr_result, polygons
            )
            # 如果有识别框没有ocr结果，直接进行rec补充
            cell_box_det_map = self.re_rec(img, polygons, cell_box_det_map, rec_again)
            # 转换为中间格式，修正识别框坐标,将物理识别框，逻辑识别框，ocr识别框整合为dict，方便后续处理
            t_rec_ocr_list = self.transform_res(cell_box_det_map, polygons, logi_points)
            # 将每个单元格中的ocr识别结果排序和同行合并，输出的html能完整保留文字的换行格式
            t_rec_ocr_list = self.sort_and_gather_ocr_res(t_rec_ocr_list)
            # cell_box_map =
            logi_points = [t_box_ocr["t_logic_box"] for t_box_ocr in t_rec_ocr_list]
            cell_box_det_map = {
                i: [ocr_box_and_text[1] for ocr_box_and_text in t_box_ocr["t_ocr_res"]]
                for i, t_box_ocr in enumerate(t_rec_ocr_list)
            }
            table_str = self.plot_html_table(logi_points, cell_box_det_map)
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

    def sort_and_gather_ocr_res(self, res):
        for i, dict_res in enumerate(res):
            _, sorted_idx = sorted_ocr_boxes(
                [ocr_det[0] for ocr_det in dict_res["t_ocr_res"]], threhold=0.3
            )
            dict_res["t_ocr_res"] = [dict_res["t_ocr_res"][i] for i in sorted_idx]
            dict_res["t_ocr_res"] = self.gather_ocr_list_by_row(
                dict_res["t_ocr_res"], threhold=0.3
            )
        return res

    def gather_ocr_list_by_row(
        self, ocr_list: List[Any], threhold: float = 0.2
    ) -> List[Any]:
        """
        :param ocr_list: [[[xmin,ymin,xmax,ymax], text]]
        :return:
        """
        threshold = 10
        for i in range(len(ocr_list)):
            if not ocr_list[i]:
                continue

            for j in range(i + 1, len(ocr_list)):
                if not ocr_list[j]:
                    continue
                cur = ocr_list[i]
                next = ocr_list[j]
                cur_box = cur[0]
                next_box = next[0]
                c_idx = is_single_axis_contained(
                    cur[0], next[0], axis="y", threhold=threhold
                )
                if c_idx:
                    dis = max(next_box[0] - cur_box[2], 0)
                    blank_str = int(dis / threshold) * " "
                    cur[1] = cur[1] + blank_str + next[1]
                    xmin = min(cur_box[0], next_box[0])
                    xmax = max(cur_box[2], next_box[2])
                    ymin = min(cur_box[1], next_box[1])
                    ymax = max(cur_box[3], next_box[3])
                    cur_box[0] = xmin
                    cur_box[1] = ymin
                    cur_box[2] = xmax
                    cur_box[3] = ymax
                    ocr_list[j] = None
        ocr_list = [x for x in ocr_list if x]
        return ocr_list

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
            crop_img = self.get_rotate_crop_image(img, sorted_polygons[i])
            pad_img = cv2.copyMakeBorder(
                crop_img, 5, 5, 100, 100, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            rec_res, _ = self.ocr(pad_img, use_det=False, use_cls=True, use_rec=True)
            box = sorted_polygons[i]
            text = [rec[0] for rec in rec_res]
            scores = [rec[1] for rec in rec_res]
            cell_box_map[i] = [[box, "".join(text), min(scores)]]
        return cell_box_map

    def match_ocr_cell(
        self, dt_rec_boxes: List[List[Union[Any, str]]], pred_bboxes: np.ndarray
    ):
        """
        :param dt_rec_boxes: [[(4.2), text, score]]
        :param pred_bboxes: shap (4,2)
        :return:
        """
        matched = {}
        not_match_orc_boxes = []
        for i, gt_box in enumerate(dt_rec_boxes):
            for j, pred_box in enumerate(pred_bboxes):
                pred_box = [
                    pred_box[0][0],
                    pred_box[0][1],
                    pred_box[2][0],
                    pred_box[2][1],
                ]
                ocr_boxes = gt_box[0]
                # xmin,ymin,xmax,ymax
                ocr_box = (
                    ocr_boxes[0][0],
                    ocr_boxes[0][1],
                    ocr_boxes[2][0],
                    ocr_boxes[2][1],
                )
                contained = is_box_contained(ocr_box, pred_box, 0.6)
                if contained == 1 or calculate_iou(ocr_box, pred_box) > 0.8:
                    if j not in matched:
                        matched[j] = [gt_box]
                    else:
                        matched[j].append(gt_box)
                else:
                    not_match_orc_boxes.append(gt_box)

        return matched, not_match_orc_boxes

    def get_rotate_crop_image(self, img: np.ndarray, points: np.ndarray) -> np.ndarray:
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3]),
            )
        )
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2]),
            )
        )
        pts_std = np.float32(
            [
                [0, 0],
                [img_crop_width, 0],
                [img_crop_width, img_crop_height],
                [0, img_crop_height],
            ]
        )
        M = cv2.getPerspectiveTransform(
            points.astype(np.float32), pts_std.astype(np.float32)
        )
        dst_img = cv2.warpPerspective(
            img,
            M,
            (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def plot_html_table(
        self,
        logi_points: Union[Union[np.ndarray, List]],
        cell_box_map: Dict[int, List[str]],
    ) -> str:
        # 初始化最大行数和列数
        max_row = 0
        max_col = 0
        # 计算最大行数和列数
        for point in logi_points:
            max_row = max(max_row, point[1] + 1)  # 加1是因为结束下标是包含在内的
            max_col = max(max_col, point[3] + 1)  # 加1是因为结束下标是包含在内的

        # 创建一个二维数组来存储 sorted_logi_points 中的元素
        grid = [[None] * max_col for _ in range(max_row)]

        valid_start_row = (1 << 16) - 1
        valid_start_col = (1 << 16) - 1
        valid_end_col = 0
        # 将 sorted_logi_points 中的元素填充到 grid 中
        for i, logic_point in enumerate(logi_points):
            row_start, row_end, col_start, col_end = (
                logic_point[0],
                logic_point[1],
                logic_point[2],
                logic_point[3],
            )
            ocr_rec_text_list = cell_box_map.get(i)
            if ocr_rec_text_list and "".join(ocr_rec_text_list):
                valid_start_row = min(row_start, valid_start_row)
                valid_start_col = min(col_start, valid_start_col)
                valid_end_col = max(col_end, valid_end_col)
            for row in range(row_start, row_end + 1):
                for col in range(col_start, col_end + 1):
                    grid[row][col] = (i, row_start, row_end, col_start, col_end)

        # 创建表格
        table_html = "<html><body><table>"

        # 遍历每行
        for row in range(max_row):
            if row < valid_start_row:
                continue
            temp = "<tr>"
            # 遍历每一列
            for col in range(max_col):
                if col < valid_start_col or col > valid_end_col:
                    continue
                if not grid[row][col]:
                    temp += "<td></td>"
                else:
                    i, row_start, row_end, col_start, col_end = grid[row][col]
                    if not cell_box_map.get(i):
                        continue
                    if row == row_start and col == col_start:
                        ocr_rec_text = cell_box_map.get(i)
                        text = "<br>".join(ocr_rec_text)
                        # 如果是起始单元格
                        row_span = row_end - row_start + 1
                        col_span = col_end - col_start + 1
                        cell_content = (
                            f"<td rowspan={row_span} colspan={col_span}>{text}</td>"
                        )
                        temp += cell_content

            table_html = table_html + temp + "</tr>"

        table_html += "</table></body></html>"
        return table_html


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
