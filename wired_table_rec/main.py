# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import argparse
import importlib
import logging
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple, Union

from .table_line_rec import TableLineRecognition
from .table_recover import TableRecover
from .utils import InputType, LoadImage
from .utils_table_recover import match_ocr_cell, plot_html_table

cur_dir = Path(__file__).resolve().parent
default_model_path = cur_dir / "models" / "cycle_center_net_v1.onnx"


class WiredTableRecognition:
    def __init__(self, table_model_path: Union[str, Path] = default_model_path):
        self.load_img = LoadImage()
        self.table_line_rec = TableLineRecognition(str(table_model_path))
        self.table_recover = TableRecover()

        try:
            self.ocr = importlib.import_module("rapidocr_onnxruntime").RapidOCR()
        except ModuleNotFoundError:
            self.ocr = None

    def __call__(
        self,
        img: InputType,
        ocr_result: Optional[List[Union[List[List[float]], str, str]]] = None,
    ) -> Tuple[str, float]:
        if self.ocr is None and ocr_result is None:
            raise ValueError(
                "One of two conditions must be met: ocr_result is not empty, or rapidocr_onnxruntime is installed."
            )

        s = time.perf_counter()

        img = self.load_img(img)
        polygons = self.table_line_rec(img)
        if polygons is None:
            logging.warning("polygons is None.")
            return "", 0.0

        try:
            table_res = self.table_recover(polygons)

            if ocr_result is None:
                ocr_result, _ = self.ocr(img)

            cell_box_map = match_ocr_cell(polygons, ocr_result)
            table_str = plot_html_table(table_res, cell_box_map)
            elapse = time.perf_counter() - s
        except Exception:
            logging.warning(traceback.format_exc())
            return "", 0.0
        else:
            return table_str, elapse


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
