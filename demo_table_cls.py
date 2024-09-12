# -*- encoding: utf-8 -*-
from table_cls import TableCls

table_cls = TableCls()
img_path = "tests/test_files/table_cls/lineless_table.png"
cls_str, elapse = table_cls(img_path)
print(cls_str)
print(elapse)
