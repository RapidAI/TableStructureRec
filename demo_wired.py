# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path

from wired_table_rec import WiredTableRecognition

table_rec = WiredTableRecognition()

img_path = "tests/test_files/wired/squeeze_error.jpeg"
table_str, elapse = table_rec(img_path)
print(table_str)
print(elapse)

with open(f"{Path(img_path).stem}.html", "w", encoding="utf-8") as f:
    f.write(table_str)
