# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

import pytest
from bs4 import BeautifulSoup
from rapidocr_onnxruntime import RapidOCR

cur_dir = Path(__file__).resolve().parent
root_dir = cur_dir.parent

sys.path.append(str(root_dir))

from wired_table_rec import WiredTableRecognition

test_file_dir = cur_dir / "test_files" / "wired"

table_recog = WiredTableRecognition()
ocr_engine = RapidOCR()


def get_td_nums(html: str) -> int:
    soup = BeautifulSoup(html, "html.parser")
    tds = soup.table.find_all("td")
    return len(tds)


def test_squeeze_bug():
    img_path = test_file_dir / "squeeze_error.jpeg"
    ocr_result, _ = ocr_engine(img_path)
    table_str, _ = table_recog(str(img_path), ocr_result)
    td_nums = get_td_nums(table_str)
    assert td_nums == 153


@pytest.mark.parametrize(
    "img_path, gt_td_nums, gt2",
    [
        ("table_recognition.jpg", 35, "d colsp"),
        ("table2.jpg", 22, "td><td "),
        ("row_span.png", 17, "></td><"),
    ],
)
def test_input_normal(img_path, gt_td_nums, gt2):
    img_path = test_file_dir / img_path

    ocr_result, _ = ocr_engine(img_path)
    table_str, _ = table_recog(str(img_path), ocr_result)
    td_nums = get_td_nums(table_str)

    assert td_nums == gt_td_nums
    assert table_str[-53:-46] == gt2

