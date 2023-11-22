# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

import cv2
import pytest

cur_dir = Path(__file__).resolve().parent
root_dir = cur_dir.parent

sys.path.append(str(root_dir))

from lineless_table_rec import LinelessTableRecognition

test_file_dir = cur_dir / "test_files"

table_recog = LinelessTableRecognition()


@pytest.mark.parametrize(
    "img_path, table_str_len, td_nums",
    [
        ("lineless_table_recognition.jpg", 2076, 108),
        ("table.jpg", 3208, 160),
    ],
)
def test_input_normal(img_path, table_str_len, td_nums):
    img_path = test_file_dir / img_path
    img = cv2.imread(str(img_path))

    table_str, _ = table_recog(img)

    assert len(table_str) >= table_str_len
    assert table_str.count("td") == td_nums
