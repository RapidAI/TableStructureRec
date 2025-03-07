# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path

from rapidocr_onnxruntime import RapidOCR

from wired_table_rec import WiredTableRecognition
from wired_table_rec.main import RapidTableInput, ModelType
from wired_table_rec.utils.utils import VisTable

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)
input_args = RapidTableInput(model_type=ModelType.CYCLE_CENTER_NET.value)
table_engine = WiredTableRecognition(input_args)
ocr_engine = RapidOCR()
viser = VisTable()
if __name__ == "__main__":
    img_path = "tests/test_files/wired/bad_case_1.png"

    ocr_result, _ = ocr_engine(img_path)
    boxes, txts, scores = list(zip(*ocr_result))

    # Table Rec
    table_results = table_engine(img_path)
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
