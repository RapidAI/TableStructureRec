# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

import pytest
from rapidocr_onnxruntime import RapidOCR

cur_dir = Path(__file__).resolve().parent
root_dir = cur_dir.parent

sys.path.append(str(root_dir))

from wired_table_rec import WiredTableRecognition

test_file_dir = cur_dir / "test_files" / "wired"

table_recog = WiredTableRecognition()
ocr_engine = RapidOCR()


@pytest.mark.parametrize(
    "img_path, gt1, gt2",
    [
        ("table_recognition.jpg", 1245, "d colsp"),
        ("table2.jpg", 924, "td><td "),
        ("row_span.png", 312, "></td><"),
    ],
)
def test_input_normal(img_path, gt1, gt2):
    img_path = test_file_dir / img_path

    ocr_result, _ = ocr_engine(img_path)
    table_str, _ = table_recog(str(img_path), ocr_result)

    assert len(table_str) >= gt1
    assert table_str[-53:-46] == gt2


@pytest.mark.parametrize(
    "img_path, gt1, gt2",
    [
        ("table_recognition.jpg", 1245, "d colsp"),
        ("table2.jpg", 924, "td><td "),
        ("row_span.png", 311, "></td><"),
    ],
)
def test_input_without_ocr(img_path, gt1, gt2):
    img_path = test_file_dir / img_path

    table_str, _ = table_recog(str(img_path))

    assert len(table_str) >= gt1
    assert table_str[-53:-46] == gt2
