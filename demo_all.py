from table_cls import TableCls
from wired_table_rec.main import WiredTableInput, WiredTableRecognition
from lineless_table_rec.main import LinelessTableInput, LinelessTableRecognition

if __name__ == "__main__":
    # Init
    wired_input = WiredTableInput()
    lineless_input = LinelessTableInput()
    wired_engine = WiredTableRecognition(wired_input)
    lineless_engine = LinelessTableRecognition(lineless_input)
    # 默认小yolo模型(0.1s)，可切换为精度更高yolox(0.25s),更快的qanything(0.07s)模型或paddle模型(0.03s)
    table_cls = TableCls()
    img_path = f"tests/test_files/table.jpg"

    cls, elasp = table_cls(img_path)
    if cls == "wired":
        table_engine = wired_engine
    else:
        table_engine = lineless_engine

    table_results = table_engine(img_path, enhance_box_line=False)
    # 使用RapidOCR输入
    # ocr_engine = RapidOCR()
    # ocr_result, _ = ocr_engine(img_path)
    # table_results = table_engine(img_path, ocr_result=ocr_result)

    # Visualize table rec result
    # save_dir = Path("outputs")
    # save_dir.mkdir(parents=True, exist_ok=True)
    #
    # save_html_path = f"outputs/{Path(img_path).stem}.html"
    # save_drawed_path = f"outputs/{Path(img_path).stem}_table_vis{Path(img_path).suffix}"
    # save_logic_path = (
    #     f"outputs/{Path(img_path).stem}_table_vis_logic{Path(img_path).suffix}"
    # )

    #
    # vis_table = VisTable()
    # vis_imged = vis_table(
    #     img_path, table_results, save_html_path, save_drawed_path, save_logic_path
    # )
