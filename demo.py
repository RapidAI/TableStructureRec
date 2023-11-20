# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path

from lineless_table_rec import LinelessTableRecognition

engine = LinelessTableRecognition()
img_path = "tests/test_files/test.jpg"
img_name = Path(img_path).stem

table_str, elapse = engine(img_path)

print(table_str)
print(elapse)
# with open(f"temp/{img_name}.html", "w", encoding="utf-8") as f:
#     f.write(table_str)

# print("ok")
