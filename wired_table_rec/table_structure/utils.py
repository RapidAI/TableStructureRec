from typing import Union, List, Tuple, Any
import numpy as np


def box_4_1_poly_to_box_4_2(poly_box: Union[list, np.ndarray]) -> List[List[float]]:
    xmin, ymin, xmax, ymax = tuple(poly_box)
    return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]


def box_4_2_poly_to_box_4_1(poly_box: Union[list, np.ndarray]) -> List[Any]:
    """
    将poly_box转换为box_4_1
    :param poly_box:
    :return:
    """
    return [poly_box[0][0], poly_box[0][1], poly_box[2][0], poly_box[2][1]]


def calculate_iou(
    box1: Union[np.ndarray, List], box2: Union[np.ndarray, List]
) -> float:
    """
    :param box1: Iterable [xmin,ymin,xmax,ymax]
    :param box2: Iterable [xmin,ymin,xmax,ymax]
    :return: iou: float 0-1
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    # 不相交直接退出检测
    if b1_x2 < b2_x1 or b1_x1 > b2_x2 or b1_y2 < b2_y1 or b1_y1 > b2_y2:
        return 0.0
    # 计算交集
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    i_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # 计算并集
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    u_area = b1_area + b2_area - i_area

    # 避免除零错误，如果区域小到乘积为0,认为是错误识别，直接去掉
    if u_area == 0:
        return 1
        # 检查完全包含
    iou = i_area / u_area
    return iou


def caculate_single_axis_iou(
    box1: Union[np.ndarray, List], box2: Union[np.ndarray, List], axis="x"
) -> float:
    """
    :param box1: Iterable [xmin,ymin,xmax,ymax]
    :param box2: Iterable [xmin,ymin,xmax,ymax]
    :return: iou: float 0-1
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2
    if axis == "x":
        i_min = max(b1_x1, b2_x1)
        i_max = min(b1_x2, b2_x2)
        u_area = max(b1_x2, b2_x2) - min(b1_x1, b2_x1)
    else:
        i_min = max(b1_y1, b2_y1)
        i_max = min(b1_y2, b2_y2)
        u_area = max(b1_y2, b2_y2) - min(b1_y1, b2_y1)
    i_area = max(i_max - i_min, 0)
    if u_area == 0:
        return 1
    return i_area / u_area


def is_box_contained(
    box1: Union[np.ndarray, List], box2: Union[np.ndarray, List], threshold=0.2
) -> Union[int, None]:
    """
    :param box1: Iterable [xmin,ymin,xmax,ymax]
    :param box2: Iterable [xmin,ymin,xmax,ymax]
    :return: 1: box1 is contained 2: box2 is contained None: no contain these
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    # 不相交直接退出检测
    if b1_x2 < b2_x1 or b1_x1 > b2_x2 or b1_y2 < b2_y1 or b1_y1 > b2_y2:
        return None
    # 计算box2的总面积
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)

    # 计算box1和box2的交集
    intersect_x1 = max(b1_x1, b2_x1)
    intersect_y1 = max(b1_y1, b2_y1)
    intersect_x2 = min(b1_x2, b2_x2)
    intersect_y2 = min(b1_y2, b2_y2)

    # 计算交集的面积
    intersect_area = max(0, intersect_x2 - intersect_x1) * max(
        0, intersect_y2 - intersect_y1
    )

    # 计算外面的面积
    b1_outside_area = b1_area - intersect_area
    b2_outside_area = b2_area - intersect_area

    # 计算外面的面积占box2总面积的比例
    ratio_b1 = b1_outside_area / b1_area if b1_area > 0 else 0
    ratio_b2 = b2_outside_area / b2_area if b2_area > 0 else 0

    if ratio_b1 < threshold:
        return 1
    if ratio_b2 < threshold:
        return 2
    # 判断比例是否大于阈值
    return None


def is_single_axis_contained(
    box1: Union[np.ndarray, List],
    box2: Union[np.ndarray, List],
    axis="x",
    threhold: float = 0.2,
) -> Union[int, None]:
    """
    :param box1: Iterable [xmin,ymin,xmax,ymax]
    :param box2: Iterable [xmin,ymin,xmax,ymax]
    :return: 1: box1 is contained 2: box2 is contained None: no contain these
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # 计算轴重叠大小
    if axis == "x":
        b1_area = b1_x2 - b1_x1
        b2_area = b2_x2 - b2_x1
        i_area = min(b1_x2, b2_x2) - max(b1_x1, b2_x1)
    else:
        b1_area = b1_y2 - b1_y1
        b2_area = b2_y2 - b2_y1
        i_area = min(b1_y2, b2_y2) - max(b1_y1, b2_y1)
        # 计算外面的面积
    b1_outside_area = b1_area - i_area
    b2_outside_area = b2_area - i_area

    ratio_b1 = b1_outside_area / b1_area if b1_area > 0 else 0
    ratio_b2 = b2_outside_area / b2_area if b2_area > 0 else 0
    if ratio_b1 < threhold:
        return 1
    if ratio_b2 < threhold:
        return 2
    return None


def sorted_ocr_boxes(
    dt_boxes: Union[np.ndarray, list], threhold: float = 0.2
) -> Tuple[Union[np.ndarray, list], List[int]]:
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with (xmin, ymin, xmax, ymax)
    return:
        sorted boxes(array) with (xmin, ymin, xmax, ymax)
    """
    num_boxes = len(dt_boxes)
    if num_boxes <= 0:
        return dt_boxes, []
    indexed_boxes = [(box, idx) for idx, box in enumerate(dt_boxes)]
    sorted_boxes_with_idx = sorted(indexed_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes, indices = zip(*sorted_boxes_with_idx)
    indices = list(indices)
    _boxes = [dt_boxes[i] for i in indices]
    threahold = 20
    # 避免输出和输入格式不对应，与函数功能不符合
    if isinstance(dt_boxes, np.ndarray):
        _boxes = np.array(_boxes)
    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            c_idx = is_single_axis_contained(
                _boxes[j], _boxes[j + 1], axis="y", threhold=threhold
            )
            if (
                c_idx is not None
                and _boxes[j + 1][0] < _boxes[j][0]
                and abs(_boxes[j][1] - _boxes[j + 1][1]) < threahold
            ):
                _boxes[j], _boxes[j + 1] = _boxes[j + 1].copy(), _boxes[j].copy()
                indices[j], indices[j + 1] = indices[j + 1], indices[j]
            else:
                break
    return _boxes, indices
