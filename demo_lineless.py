# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path

from lineless_table_rec import LinelessTableRecognition
from lineless_table_rec.utils_table_recover import (
    format_html,
    plot_rec_box,
    plot_rec_box_with_logic_info,
)

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

img_path = "tests/test_files/lineless_table_recognition.jpg"
table_rec = LinelessTableRecognition()

html, elasp, polygons, logic_points, ocr_res = table_rec(img_path)
print(f"cost: {elasp:.5f}")

complete_html = format_html(html)

save_table_path = output_dir / "table.html"
with open(save_table_path, "w", encoding="utf-8") as file:
    file.write(complete_html)

plot_rec_box_with_logic_info(
    img_path, f"{output_dir}/table_rec_box.jpg", logic_points, polygons
)
plot_rec_box(img_path, f"{output_dir}/ocr_box.jpg", ocr_res)
print(f"The results has been saved under {output_dir}")
