# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import logging
import time
import traceback
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Union, Optional, Any

import numpy as np

from .table_structure_lore import TSRLore
from .utils.download_model import DownloadModel
from .utils.utils import InputType, LoadImage
from lineless_table_rec.utils.utils_table_recover import (
    box_4_2_poly_to_box_4_1,
    filter_duplicated_box,
    gather_ocr_list_by_row,
    match_ocr_cell,
    plot_html_table,
    sorted_ocr_boxes,
    box_4_1_poly_to_box_4_2,
)


class ModelType(Enum):
    LORE = "lore"


ROOT_URL = "https://www.modelscope.cn/models/RapidAI/RapidTable/resolve/master/"
KEY_TO_MODEL_URL = {
    ModelType.LORE.value: {
        "lore_detect": f"{ROOT_URL}/lore/detect.onnx",
        "lore_process": f"{ROOT_URL}/lore/process.onnx",
    },
}


@dataclass
class LinelessTableInput:
    model_type: Optional[str] = ModelType.LORE.value
    model_path: Union[str, Path, None, Dict[str, str]] = None
    use_cuda: bool = False
    device: str = "cpu"


@dataclass
class LinelessTableOutput:
    pred_html: Optional[str] = None
    cell_bboxes: Optional[np.ndarray] = None
    logic_points: Optional[np.ndarray] = None
    elapse: Optional[float] = None


class LinelessTableRecognition:
    def __init__(self, config: LinelessTableInput):
        self.model_type = config.model_type
        if self.model_type not in KEY_TO_MODEL_URL:
            model_list = ",".join(KEY_TO_MODEL_URL)
            raise ValueError(
                f"{self.model_type} is not supported. The currently supported models are {model_list}."
            )

        config.model_path = self.get_model_path(config.model_type, config.model_path)
        self.table_structure = TSRLore(asdict(config))
        self.load_img = LoadImage()

    def __call__(
        self,
        content: InputType,
        ocr_result: Optional[List[Union[List[List[float]], str, str]]] = None,
        **kwargs,
    ) -> LinelessTableOutput:
        s = time.perf_counter()
        need_ocr = True
        if kwargs:
            need_ocr = kwargs.get("need_ocr", True)
        img = self.load_img(content)
        try:
            polygons, logi_points = self.table_structure(img)
            if not need_ocr:
                sorted_polygons, idx_list = sorted_ocr_boxes(
                    [box_4_2_poly_to_box_4_1(box) for box in polygons]
                )
                return LinelessTableOutput(
                    "",
                    sorted_polygons,
                    logi_points[idx_list],
                    time.perf_counter() - s,
                )

            # ocr 结果匹配
            cell_box_det_map, no_match_ocr_det = match_ocr_cell(ocr_result, polygons)
            # 如果有识别框没有ocr结果，直接进行rec补充
            cell_box_det_map = self.fill_blank_rec(img, polygons, cell_box_det_map)
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
            # 将同一个识别框中的ocr结果排序并同行合并
            t_rec_ocr_list = self.sort_and_gather_ocr_res(t_rec_ocr_list)
            # 渲染为html
            polygons = [
                box_4_1_poly_to_box_4_2(t_box_ocr["t_box"])
                for t_box_ocr in t_rec_ocr_list
            ]
            logi_points = [t_box_ocr["t_logic_box"] for t_box_ocr in t_rec_ocr_list]
            cell_box_det_map = {
                i: [ocr_box_and_text[1] for ocr_box_and_text in t_box_ocr["t_ocr_res"]]
                for i, t_box_ocr in enumerate(t_rec_ocr_list)
            }
            pred_html = plot_html_table(logi_points, cell_box_det_map)

            # 输出可视化排序,用于验证结果，生产版本可以去掉
            _, idx_list = sorted_ocr_boxes(
                [t_box_ocr["t_box"] for t_box_ocr in t_rec_ocr_list]
            )
            polygons = np.array(polygons).reshape(-1, 8)
            logi_points = np.array(logi_points)
            elapse = time.perf_counter() - s
        except Exception:
            logging.warning(traceback.format_exc())
            return LinelessTableOutput("", None, None, 0.0)
        return LinelessTableOutput(pred_html, polygons, logi_points, elapse)

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

    @staticmethod
    def get_model_path(
        model_type: str, model_path: Union[str, Path, None]
    ) -> Union[str, Dict[str, str]]:
        if model_path is not None:
            return model_path

        model_url = KEY_TO_MODEL_URL.get(model_type, None)
        if isinstance(model_url, str):
            model_path = DownloadModel.download(model_url)
            return model_path

        if isinstance(model_url, dict):
            model_paths = {}
            for k, url in model_url.items():
                model_paths[k] = DownloadModel.download(
                    url, save_model_name=f"{model_type}_{Path(url).name}"
                )
            return model_paths

        raise ValueError(f"Model URL: {type(model_url)} is not between str and dict.")

    def sort_and_gather_ocr_res(self, res):
        for i, dict_res in enumerate(res):
            _, sorted_idx = sorted_ocr_boxes(
                [ocr_det[0] for ocr_det in dict_res["t_ocr_res"]], threhold=0.3
            )
            dict_res["t_ocr_res"] = [dict_res["t_ocr_res"][i] for i in sorted_idx]
            dict_res["t_ocr_res"] = gather_ocr_list_by_row(
                dict_res["t_ocr_res"], thehold=0.3
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

    def fill_blank_rec(
        self,
        img: np.ndarray,
        sorted_polygons: np.ndarray,
        cell_box_map: Dict[int, List[str]],
    ) -> Dict[int, List[Any]]:
        """找到poly对应为空的框，尝试将直接将poly框直接送到识别中"""
        for i in range(sorted_polygons.shape[0]):
            if cell_box_map.get(i):
                continue
            box = sorted_polygons[i]
            cell_box_map[i] = [[box, "", 1]]
            continue
        return cell_box_map
