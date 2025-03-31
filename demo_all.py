from wired_table_rec.utils.utils import VisTable
from table_cls import TableCls
from wired_table_rec.main import WiredTableInput, WiredTableRecognition
from lineless_table_rec.main import LinelessTableInput, LinelessTableRecognition
from rapidocr import RapidOCR

if __name__ == "__main__":
    # Init
    wired_input = WiredTableInput()
    lineless_input = LinelessTableInput()
    wired_engine = WiredTableRecognition(wired_input)
    lineless_engine = LinelessTableRecognition(lineless_input)
    viser = VisTable()
    # 默认小yolo模型(0.1s)，可切换为精度更高yolox(0.25s),更快的qanything(0.07s)模型或paddle模型(0.03s)
    table_cls = TableCls()
    img_path = f"tests/test_files/table.jpg"

    cls, elasp = table_cls(img_path)
    if cls == "wired":
        table_engine = wired_engine
    else:
        table_engine = lineless_engine

    # 使用RapidOCR输入
    ocr_engine = RapidOCR()
    rapid_ocr_output = ocr_engine(img_path, return_word_box=True)
    ocr_result = list(
        zip(rapid_ocr_output.boxes, rapid_ocr_output.txts, rapid_ocr_output.scores)
    )
    table_results = table_engine(img_path, ocr_result=ocr_result)

    # 使用单字识别
    # word_results = rapid_ocr_output.word_results
    # ocr_result = [
    #     [word_result[2], word_result[0], word_result[1]] for word_result in word_results
    # ]
    # table_results = table_engine(
    #     img_path, ocr_result=ocr_result, enhance_box_line=False
    # )

    # Save
    # save_dir = Path("outputs")
    # save_dir.mkdir(parents=True, exist_ok=True)
    #
    # save_html_path = f"outputs/{Path(img_path).stem}.html"
    # save_drawed_path = f"outputs/{Path(img_path).stem}_table_vis{Path(img_path).suffix}"
    # save_logic_path = (
    #     f"outputs/{Path(img_path).stem}_table_vis_logic{Path(img_path).suffix}"
    # )

    # Visualize table rec result
    # vis_imged = viser(
    #     img_path, table_results, save_html_path, save_drawed_path, save_logic_path
    # )
