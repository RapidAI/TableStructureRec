import pytest
import numpy as np
from wired_table_rec.utils_table_line_rec import (
    _order_points,
    calculate_center_rotate_angle,
    fit_line,
    line_to_line,
    min_area_rect,
    adjust_lines,
)


@pytest.mark.parametrize(
    "pts, expected",
    [
        # 顺时针顺序正确，无需排序
        (
            np.array([[10, 10], [20, 10], [20, 20], [10, 20]]),
            np.array([[10, 10], [20, 10], [20, 20], [10, 20]], dtype="float32"),
        ),
        # 完全相反顺序，进行重排序
        (
            np.array([[20, 10], [20, 20], [10, 20], [10, 10]]),
            np.array([[10, 10], [20, 10], [20, 20], [10, 20]], dtype="float32"),
        ),
        # 部分错位顺序，重排序
        (
            np.array([[10, 20], [20, 20], [20, 10], [10, 10]]),
            np.array([[10, 10], [20, 10], [20, 20], [10, 20]], dtype="float32"),
        ),
    ],
)
def test_order_points(pts, expected):
    """
    排序后得到[(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)]
    """
    result = _order_points(pts)
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "box, expected_angle, expected_w, expected_h, expected_cx, expected_cy",
    [
        # 沿中心点无旋转
        ([10, 10, 20, 10, 20, 20, 10, 20], 0.0, 10.0, 10.0, 15.0, 15.0),
        # 沿中心点有旋转30度
        (
            [
                13.16987,
                8.1698,
                21.830,
                13.16987,
                16.830127018922195,
                21.83012701892219,
                8.169872981077807,
                16.830127018922195,
            ],
            np.pi / 6,
            10.0,
            10.0,
            15.0,
            15.0,
        ),
    ],
)
def test_calculate_center_rotate_angle(
    box, expected_angle, expected_w, expected_h, expected_cx, expected_cy
):
    angle, w, h, cx, cy = calculate_center_rotate_angle(box)
    assert np.isclose(angle, expected_angle, atol=1e-5)
    assert np.isclose(w, expected_w, atol=1e-5)
    assert np.isclose(h, expected_h, atol=1e-5)
    assert np.isclose(cx, expected_cx, atol=1e-5)
    assert np.isclose(cy, expected_cy, atol=1e-5)


# 测试函数
@pytest.mark.parametrize(
    "points, expected_A, expected_B, expected_C",
    [
        # 根据两个点计算直线方程的参数
        ([(0, 0), (1, 1)], 1, -1, 0)
    ],
)
def test_fit_line(points, expected_A, expected_B, expected_C):
    A, B, C = fit_line(points)
    assert np.isclose(A, expected_A, atol=1e-5)
    assert np.isclose(B, expected_B, atol=1e-5)
    assert np.isclose(C, expected_C, atol=1e-5)


@pytest.mark.parametrize(
    "points1, points2, expected_result",
    [
        # 横线在竖线同边，无角度偏移，延长第二个点到相交点
        ([0, 0, 0.9, 0], [1, 0, 1, 1], np.array([0, 0, 1, 0], dtype="float32")),
        # 横线在竖线同边，有角度偏移，延长第一个点到相交点
        ([4, 3, 0, 0], [8, 0, 8, 8], np.array([8, 6, 0, 0], dtype="float32")),
        # 横线在竖线异边，不进行延伸
        ([0, 0, 2, 1], [1, 0, 1, 1], np.array([0, 0, 2, 1], dtype="float32")),
        # 超过偏移角度，不进行延伸
        ([0, 0, 0.9, 0.9], [1, 0, 1, 4], np.array([0, 0, 0.9, 0.9], dtype="float32")),
        # 超过交点绝对值长度，不进行延伸
        ([4, 3, 0, 0], [50, 0, 50, 50], np.array([4, 3, 0, 0], dtype="float32"))
        #
    ],
)
def test_line_to_line(points1, points2, expected_result):
    # 为测试方便，提高角度阈值到60度
    result = line_to_line(points1, points2, angle=38)
    assert np.allclose(result, expected_result, atol=1e-5)


@pytest.mark.parametrize(
    "coords, expected_result",
    [
        # 竖线求最小外接矩形
        (
            np.array([[0, 1000], [10, 1000], [10, 1002], [20, 1002]]),
            [1000, 0, 1002, 20],
        ),
        # 横线求最小外接矩形
        (
            np.array([[1000, 0], [1000, 10], [1002, 15], [1001, 30]]),
            [0, 1000, 30, 1000],
        ),
    ],
)
def test_min_area_rect(coords, expected_result):
    result = min_area_rect(coords)
    assert np.allclose(result, expected_result, atol=2)


@pytest.mark.parametrize(
    "lines, alph, angle, expected_result",
    [
        # 每个坐标点都能合并
        (
            [(0, 0, 1, 0), (1, 0, 2, 0)],
            # alph: 最大允许距离
            50,
            # angle: 角度阈值
            50,
            # 预期结果：两两合并
            [
                (0, 0, 1, 0),
                (0, 0, 2, 0),
                (1, 0, 1, 0),
                (1, 0, 2, 0),
                (1, 0, 0, 0),
                (1, 0, 1, 0),
                (2, 0, 0, 0),
                (2, 0, 1, 0),
            ],
        ),
        # y轴重叠过大不合并
        (
            [(0, 0.5, 0, 1.8), (0, 1, 0, 2)],
            # alph: 最大允许距离
            50,
            # angle: 角度阈值
            50,
            [],
        ),
        # x轴重叠过大不合并
        (
            [(1, 0, 2, 0), (0, 0, 1.8, 0)],
            # alph: 最大允许距离
            50,
            # angle: 角度阈值
            50,
            [],
        ),
        # 距离超过阈值不合并
        (
            [(0, 0, 1, 0), (11, 0, 13, 0)],
            # alph: 最大允许距离
            10,
            # angle: 角度阈值
            50,
            ([]),
        ),
        # 角度超过阈值不合并
        (
            # 横线距离足够近
            [(0, 0, 1, 1), (1, 1, 2, 2), (2, 2, 3, 3)],
            # alph: 最大允许距离
            100,
            # angle: 角度阈值
            35,
            # 预期结果：只有边界角度为0能合并
            ([(1, 1, 1, 1), (1, 1, 1, 1), (2, 2, 2, 2), (2, 2, 2, 2)]),
        ),
        # 多段合并,角度过滤，距离过滤同时存在，且有可以合并的点
        (
            [(0, 0, 1, 1), (1, 1, 2, 2), (2, 2, 100, 100)],
            # alph: 最大允许距离
            50,
            # angle: 角度阈值
            30,
            # 预期结果：多条竖线合并为一条线
            ([(1, 1, 1, 1), (1, 1, 1, 1), (2, 2, 2, 2), (2, 2, 2, 2)]),
        ),
        # 只有一条线
        (
            [(0, 0, 1, 0)],
            # alph: 最大允许距离
            50,
            # angle: 角度阈值
            50,
            # 预期结果：横线不变
            ([]),
        ),
    ],
)
def test_adjust_lines(lines, alph, angle, expected_result):
    result = adjust_lines(lines, alph, angle)
    assert result == expected_result
