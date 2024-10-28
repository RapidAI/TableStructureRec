import sys
from pathlib import Path

import pytest

from table_cls import TableCls

cur_dir = Path(__file__).resolve().parent
root_dir = cur_dir.parent

sys.path.append(str(root_dir))
test_file_dir = cur_dir / "test_files" / "table_cls"
table_cls = TableCls()


@pytest.mark.parametrize(
    "img_path, expected",
    [("wired_table.jpg", "wired"), ("lineless_table.png", "wireless")],
)
def test_input_normal(img_path, expected):
    img_path = test_file_dir / img_path
    res, elasp = table_cls(img_path)
    assert res == expected
