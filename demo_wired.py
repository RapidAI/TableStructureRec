# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path

from wired_table_rec import WiredTableRecognition
from wired_table_rec.utils_table_recover import (
    format_html,
    plot_rec_box,
    plot_rec_box_with_logic_info,
)

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

table_rec = WiredTableRecognition()

img_path = "tests/test_files/wired/wired_big_box.png"
html, elasp, polygons, logic_points, ocr_res = table_rec(
    img_path,
    version="v2",  # 默认使用v2线框模型，切换阿里读光模型可改为v1
    morph_close=True,  # 是否进行形态学操作,辅助找到更多线框,默认为True
    more_h_lines=True,  # 是否基于线框检测结果进行更多水平线检查，辅助找到更小线框, 默认为True
    more_v_lines=True,  # 是否基于线框检测结果进行更多垂直线检查，辅助找到更小线框, 默认为True
    extend_line=True,  # 是否基于线框检测结果进行线段延长，辅助找到更多线框, 默认为True
    need_ocr=True,  # 是否进行OCR识别, 默认为True
    rec_again=True,  # 是否针对未识别到文字的表格框,进行单独截取再识别,默认为True
)

print(f"cost: {elasp:.5f}")

complete_html = format_html(html)

save_table_path = output_dir / "table.html"
with open(save_table_path, "w", encoding="utf-8") as file:
    file.write(complete_html)

plot_rec_box_with_logic_info(
    img_path, f"{output_dir}/table_rec_box.jpg", logic_points, polygons
)
plot_rec_box(f"{output_dir}/table_rec_box.jpg", f"{output_dir}/ocr_box.jpg", ocr_res)

print(f"The results has been saved under {output_dir}")
