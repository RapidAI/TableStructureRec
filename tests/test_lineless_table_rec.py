# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path
import pytest

cur_dir = Path(__file__).resolve().parent
root_dir = cur_dir.parent

sys.path.append(str(root_dir))

from lineless_table_rec.utils_table_recover import *
from lineless_table_rec import LinelessTableRecognition

test_file_dir = cur_dir / "test_files"

table_recog = LinelessTableRecognition()


@pytest.mark.parametrize(
    "img_path, table_str_len, td_nums",
    [
        ("lineless_table_recognition.jpg", 1869, 104),
        ("table.jpg", 3000, 158),
    ],
)
def test_input_normal(img_path, table_str_len, td_nums):
    img_path = test_file_dir / img_path
    img = cv2.imread(str(img_path))

    table_str, *_ = table_recog(img)

    assert len(table_str) >= table_str_len
    assert table_str.count("td") == td_nums


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
    "table_boxes, expected_delete_idx",
    [
        # 去除包含和重叠的盒子
        (
            np.array(
                [
                    [10, 20, 30, 40],
                    [10, 20, 30, 40],
                    [10, 30, 30, 40],
                    [9, 35, 25, 50],
                    [10, 19, 29, 41],
                ]
            ),
            {1, 2, 4},
        ),
        # 一个盒子的情况
        (np.array([[1, 2, 3, 4]]), set()),
        # 无盒子的情况
        (np.array([]), set()),
    ],
)
def test_filter_duplicated_box(table_boxes, expected_delete_idx):
    delete_idx = filter_duplicated_box(table_boxes.tolist())
    assert (
        delete_idx == expected_delete_idx
    ), f"Expected {expected_delete_idx}, but got {delete_idx}"
