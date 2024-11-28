# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path
import numpy as np
import pytest
from bs4 import BeautifulSoup
from rapidocr_onnxruntime import RapidOCR

from wired_table_rec.utils import rescale_size
from wired_table_rec.utils_table_recover import (
    plot_html_table,
    is_single_axis_contained,
    gather_ocr_list_by_row,
    sorted_ocr_boxes,
    is_box_contained,
)

cur_dir = Path(__file__).resolve().parent
root_dir = cur_dir.parent

sys.path.append(str(root_dir))

from wired_table_rec import WiredTableRecognition

test_file_dir = cur_dir / "test_files" / "wired"

table_recog = WiredTableRecognition()
ocr_engine = RapidOCR()


def get_td_nums(html: str) -> int:
    soup = BeautifulSoup(html, "html.parser")
    if not soup.table:
        return 0
    tds = soup.table.find_all("td")
    return len(tds)


def test_squeeze_bug():
    img_path = test_file_dir / "squeeze_error.jpeg"
    ocr_result, _ = ocr_engine(img_path)
    table_str, *_ = table_recog(str(img_path), ocr_result)
    td_nums = get_td_nums(table_str)
    assert td_nums >= 160


@pytest.mark.parametrize(
    "img_path, gt_td_nums, gt2",
    [
        ("table_recognition.jpg", 35, "d colsp"),
        ("table2.jpg", 23, "td><td "),
        ("row_span.png", 17, "></td><"),
        ("no_table.jpg", 1, "d colsp"),
    ],
)
def test_input_normal(img_path, gt_td_nums, gt2):
    img_path = test_file_dir / img_path

    ocr_result, _ = ocr_engine(img_path)
    table_str, *_ = table_recog(str(img_path), ocr_result)
    td_nums = get_td_nums(table_str)

    assert td_nums >= gt_td_nums


@pytest.mark.parametrize(
    "img_path, gt_td_nums",
    [
        ("wired_big_box.png", 44),
    ],
)
def test_enhance_box_line(img_path, gt_td_nums):
    img_path = test_file_dir / img_path

    ocr_result, _ = ocr_engine(img_path)
    table_str, *_ = table_recog(str(img_path), ocr_result, enhance_box_line=False)
    td_nums = get_td_nums(table_str)

    assert td_nums <= gt_td_nums


@pytest.mark.parametrize(
    "box1, box2, threshold, expected",
    [
        # Box1 完全包含在 Box2 内
        ([[10, 20, 30, 40], [5, 15, 45, 55], 0.2, 1]),
        # Box2 完全包含在 Box1 内
        ([[5, 15, 45, 55], [10, 20, 30, 40], 0.2, 2]),
        # Box1 和 Box2 部分重叠，但不满足阈值
        ([[10, 20, 30, 40], [25, 35, 45, 55], 0.2, None]),
        # Box1 和 Box2 完全不重叠
        ([[10, 20, 30, 40], [50, 60, 70, 80], 0.2, None]),
        # Box1 和 Box2 有交集，但不满足阈值
        ([[10, 20, 30, 40], [15, 25, 35, 45], 0.2, None]),
        # Box1 和 Box2 有交集，且满足阈值
        ([[10, 20, 30, 40], [15, 25, 35, 45], 0.5, 1]),
        # Box1 和 Box2 有交集，且满足阈值
        ([[15, 25, 35, 45], [14, 24, 16, 44], 0.6, 2]),
        # Box1 和 Box2 相同
        ([[10, 20, 30, 40], [10, 20, 30, 40], 0.2, 1]),
        # 使用 NumPy 数组作为输入
        ([np.array([10, 20, 30, 40]), np.array([5, 15, 45, 55]), 0.2, 1]),
    ],
)
def test_is_box_contained(box1, box2, threshold, expected):
    result = is_box_contained(box1, box2, threshold)
    assert result == expected, f"Expected {expected}, but got {result}"


@pytest.mark.parametrize(
    "box1, box2, axis, threshold, expected",
    [
        # Box1 完全包含 Box2 (X轴)
        ([10, 10, 20, 20], [12, 12, 18, 18], "x", 0.2, 2),
        # Box2 完全包含 Box1 (X轴)
        ([12, 12, 18, 18], [10, 10, 20, 20], "x", 0.2, 1),
        # Box1 完全包含 Box2 (Y轴)
        ([10, 10, 20, 20], [12, 12, 18, 18], "y", 0.2, 2),
        # Box2 完全包含 Box1 (Y轴)
        ([12, 12, 18, 18], [10, 10, 20, 20], "y", 0.2, 1),
        # Box1 和 Box2 不相交 (X轴)
        ([10, 10, 20, 20], [25, 25, 30, 30], "x", 0.2, None),
        # Box1 和 Box2 不相交 (Y轴)
        ([10, 10, 20, 20], [25, 25, 30, 30], "y", 0.2, None),
        # Box1 部分包含 Box2 (X轴)-超过阈值
        ([10, 10, 20, 20], [15, 15, 25, 25], "x", 0.2, None),
        # Box1 部分包含 Box2 (Y轴)-超过阈值
        ([10, 10, 20, 20], [15, 15, 25, 25], "y", 0.2, None),
        # Box1 部分包含 Box2 (X轴)-满足阈值
        ([10, 10, 20, 20], [13, 15, 21, 25], "x", 0.2, 2),
        # Box2 部分包含 Box1 (Y轴)-满足阈值
        ([10, 14, 20, 20], [15, 15, 25, 50], "y", 0.2, 1),
        # Box1 和 Box2 完全重合 (X轴)
        ([10, 10, 20, 20], [10, 10, 20, 20], "x", 0.2, 1),
        # Box1 和 Box2 完全重合 (Y轴)
        ([10, 10, 20, 20], [10, 10, 20, 20], "y", 0.2, 1),
    ],
)
def test_is_single_axis_contained(box1, box2, axis, threshold, expected):
    result = is_single_axis_contained(box1, box2, axis, threshold)
    assert result == expected


@pytest.mark.parametrize(
    "input_ocr_list, expected_output",
    [
        (
            [[[10, 20, 30, 40], "text1"], [[15, 23, 35, 43], "text2"]],
            [[[10, 20, 35, 43], "text1text2"]],
        ),
        (
            [
                [[10, 24, 30, 30], "text1"],
                [[15, 25, 35, 45], "text2"],
                [[5, 30, 15, 50], "text3"],
            ],
            [[[10, 24, 35, 45], "text1text2"], [[5, 30, 15, 50], "text3"]],
        ),
        ([], []),
        (
            [[[10, 20, 30, 40], "text1"], [], [[15, 25, 35, 45], "text2"]],
            [[[10, 20, 30, 40], "text1"], [[15, 25, 35, 45], "text2"]],
        ),
    ],
)
def test_gather_ocr_list_by_row(input_ocr_list, expected_output):
    result = gather_ocr_list_by_row(input_ocr_list)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"


@pytest.mark.parametrize(
    "dt_boxes, expected_boxes, expected_indices",
    [
        # 基本排序情况
        (
            np.array([[2, 3, 4, 5], [3, 4, 5, 6], [1, 2, 2, 3]]),
            np.array([[1, 2, 2, 3], [2, 3, 4, 5], [3, 4, 5, 6]]),
            [2, 0, 1],
        ),
        # 基本排序错误，修正正确
        (
            np.array([[59, 0, 148, 52], [134, 0, 254, 53], [12, 13, 30, 40]]),
            np.array([[12, 13, 30, 40], [59, 0, 148, 52], [134, 0, 254, 53]]),
            [2, 0, 1],
        ),
        # 一个盒子的情况
        (np.array([[2, 3, 4, 5]]), np.array([[2, 3, 4, 5]]), [0]),
        # 无盒子的情况
        (np.array([]), np.array([]), []),
    ],
)
def test_sorted_ocr_boxes(dt_boxes, expected_boxes, expected_indices):
    sorted_boxes, indices = sorted_ocr_boxes(dt_boxes)
    assert (
        sorted_boxes.tolist() == expected_boxes.tolist()
    ), f"Expected {expected_boxes.tolist()}, but got {sorted_boxes.tolist()}"
    assert (
        indices == expected_indices
    ), f"Expected {expected_indices}, but got {indices}"


@pytest.mark.parametrize(
    "old_size, scale, return_scale, expected_result",
    [
        # 以短边为准进行缩放
        ((100, 50), (300, 100), True, ((200, 100), 2.0)),
        ((50, 100), (100, 300), True, ((100, 200), 2.0)),
        # 以长边为准进行缩放
        ((100, 50), (200, 150), True, ((200, 100), 2.0)),
        ((50, 100), (150, 200), True, ((100, 200), 2.0)),
    ],
)
def test_rescale_size(old_size, scale, return_scale, expected_result):
    result = rescale_size(old_size, scale, return_scale)
    assert np.isclose(result[1], expected_result[1], atol=1e-5)
    assert (
        result[0] == expected_result[0]
    ), f"Expected {expected_result}, but got {result}"


@pytest.mark.parametrize(
    "logi_points, cell_box_map, expected_html",
    [
        # 测试空输入
        ([], {}, "<html><body><table></table></body></html>"),
        # 测试单个单元格，包含rowspan和colspan
        (
            [[0, 0, 0, 0]],
            {0: ["Cell 1"]},
            "<html><body><table><tr><td rowspan=1 colspan=1>Cell 1</td></tr></table></body></html>",
        ),
        # 测试多个独立单元格
        (
            [[0, 0, 0, 0], [1, 1, 1, 1]],
            {0: ["Cell 1"], 1: ["Cell 2"]},
            "<html><body><table><tr><td rowspan=1 colspan=1>Cell 1</td><td></td></tr><tr><td></td><td rowspan=1 colspan=1>Cell 2</td></tr></table></body></html>",
        ),
        # 测试跨行的单元格
        (
            [[0, 1, 0, 0]],
            {0: ["Row 1 Col 1", "Row 2 Col 1"]},
            "<html><body><table><tr><td rowspan=2 colspan=1>Row 1 Col 1<br>Row 2 Col 1</td></tr><tr></tr></table></body></html>",
        ),
        # 测试跨列的单元格
        (
            [[0, 0, 0, 1]],
            {0: ["Col 1 Row 1", "Col 2 Row 1"]},
            "<html><body><table><tr><td rowspan=1 colspan=2>Col 1 Row 1<br>Col 2 Row 1</td></tr></table></body></html>",
        ),
        # 测试跨多行多列的单元格
        (
            [[0, 1, 0, 1]],
            {0: ["Row 1 Col 1", "Row 2 Col 1"]},
            "<html><body><table><tr><td rowspan=2 colspan=2>Row 1 Col 1<br>Row 2 Col 1</td></tr><tr></tr></table></body></html>",
        ),
        # 测试跨行跨行跨列的单元格出现在中间
        (
            [[0, 0, 0, 0], [0, 1, 1, 2]],
            {0: ["Cell 1"], 1: ["Row 2", "Col 2"]},
            "<html><body><table><tr><td rowspan=1 colspan=1>Cell 1</td><td rowspan=2 colspan=2>Row 2<br>Col 2</td></tr><tr><td></td></tr></table></body></html>",
        ),
        # 测试跨行跨列的单元格出现在结尾
        (
            [[0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 2, 2]],
            {0: ["Cell 1"], 1: ["Cell 2"], 2: ["Row 1 Col 2", "Row 2 Col 2"]},
            "<html><body><table><tr><td rowspan=1 colspan=1>Cell 1</td><td></td><td rowspan=2 colspan=1>Row 1 Col 2<br>Row 2 Col 2</td></tr><tr><td></td><td rowspan=1 colspan=1>Cell 2</td></tr></table></body></html>",
        ),
        # 测试去除无效行和无效列
        (
            [[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 1, 2]],
            {2: ["Row 3 Col 1", "Row 3 Col 2"]},
            "<html><body><table><tr><td rowspan=1 colspan=1>Row 3 Col 1<br>Row 3 Col 2</td></tr></table></body></html>",
        ),
    ],
)
def test_plot_html_table(logi_points, cell_box_map, expected_html):
    html_output = plot_html_table(logi_points, cell_box_map)
    assert (
        html_output == expected_html
    ), f"Expected HTML does not match. Got: {html_output}"


@pytest.mark.parametrize(
    "img_path, gt_td_nums, gt2",
    [
        ("table_recognition.jpg", 20, "d colsp"),
    ],
)
def test_no_rec_again(img_path, gt_td_nums, gt2):
    img_path = test_file_dir / img_path

    ocr_result, _ = ocr_engine(img_path)
    table_str, *_ = table_recog(str(img_path), ocr_result, rec_again=False)
    td_nums = get_td_nums(table_str)

    assert td_nums >= gt_td_nums


@pytest.mark.parametrize(
    "img_path, html_output, points_len",
    [
        ("table2.jpg", "", 20),
        ("row_span.png", "", 14),
    ],
)
def test_no_ocr(img_path, html_output, points_len):
    img_path = test_file_dir / img_path

    ocr_result, _ = ocr_engine(img_path)
    html, elasp, polygons, logic_points, ocr_res = table_recog(
        str(img_path), ocr_result, need_ocr=False
    )
    assert len(ocr_res) == 0
    assert len(polygons) > points_len
    assert len(logic_points) > points_len
    assert len(polygons) == len(logic_points)
    assert html == html_output
