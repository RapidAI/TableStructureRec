# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import os
from lineless_table_rec.utils_table_recover import format_html
from wired_table_rec import WiredTableRecognition
from wired_table_rec.utils_table_recover import (
    plot_rec_box,
    plot_rec_box_with_logic_info,
)

output_dir = "outputs"
table_rec = WiredTableRecognition()

img_path = "tests/test_files/wired/table1.png"
html, elasp, polygons, logic_points, ocr_res = table_rec(img_path)
print(f"cost: {elasp:.5f}")
complete_html = format_html(html)
os.makedirs(os.path.dirname(f"{output_dir}/table.html"), exist_ok=True)
with open(f"{output_dir}/table.html", "w", encoding="utf-8") as file:
    file.write(complete_html)

plot_rec_box_with_logic_info(
    img_path, f"{output_dir}/table_rec_box.jpg", logic_points, polygons
)
plot_rec_box(img_path, f"{output_dir}/ocr_box.jpg", ocr_res)
