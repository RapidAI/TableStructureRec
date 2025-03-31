# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path

from rapidocr import RapidOCR

from lineless_table_rec import LinelessTableRecognition
from lineless_table_rec.main import LinelessTableInput
from lineless_table_rec.utils.utils import VisTable

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)
input_args = LinelessTableInput()
table_engine = LinelessTableRecognition(input_args)
ocr_engine = RapidOCR()
viser = VisTable()

if __name__ == "__main__":
    img_path = "tests/test_files/lineless_table_recognition.jpg"

    rapid_ocr_output = ocr_engine(img_path, return_word_box=True)
    ocr_result = list(
        zip(rapid_ocr_output.boxes, rapid_ocr_output.txts, rapid_ocr_output.scores)
    )

    # 使用单字识别
    # word_results = rapid_ocr_output.word_results
    # ocr_result = [[word_result[2], word_result[0], word_result[1]] for word_result in word_results]

    # Table Rec
    table_results = table_engine(img_path, ocr_result=ocr_result)
    table_html_str, table_cell_bboxes = (
        table_results.pred_html,
        table_results.cell_bboxes,
    )

    # Save
    save_dir = Path("outputs")
    save_dir.mkdir(parents=True, exist_ok=True)

    save_html_path = f"outputs/{Path(img_path).stem}.html"
    save_drawed_path = f"outputs/{Path(img_path).stem}_table_vis{Path(img_path).suffix}"
    save_logic_path = (
        f"outputs/{Path(img_path).stem}_table_vis_logic{Path(img_path).suffix}"
    )

    # Visualize table rec result
    vis_imged = viser(
        img_path, table_results, save_html_path, save_drawed_path, save_logic_path
    )

    print(f"The results has been saved under {output_dir}")
