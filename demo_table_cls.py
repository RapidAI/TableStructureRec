# -*- encoding: utf-8 -*-
from table_cls import TableCls

if __name__ == "__main__":
    table_cls = TableCls(model_type="yolox")
    img_path = "tests/test_files/table_cls/lineless_table_2.png"
    cls_str, elapse = table_cls(img_path)
    print(cls_str)
    print(elapse)
