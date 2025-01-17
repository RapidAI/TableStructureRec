import os

import cv2


def plot_rec_box_with_logic_info(img_path, output_path, logic_points, sorted_polygons):
    """
    :param img_path
    :param output_path
    :param logic_points: [row_start,row_end,col_start,col_end]
    :param sorted_polygons: [xmin,ymin,xmax,ymax]
    :return:
    """
    # 读取原图
    img = cv2.imread(img_path)
    img = cv2.copyMakeBorder(
        img, 0, 0, 0, 100, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    # 绘制 polygons 矩形
    for idx, polygon in enumerate(sorted_polygons):
        x0, y0, x1, y1 = polygon[0], polygon[1], polygon[2], polygon[3]
        x0 = round(x0)
        y0 = round(y0)
        x1 = round(x1)
        y1 = round(y1)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
        # 增大字体大小和线宽
        font_scale = 0.9  # 原先是0.5
        thickness = 1  # 原先是1
        logic_point = logic_points[idx]
        cv2.putText(
            img,
            f"row: {logic_point[0]}-{logic_point[1]}",
            (x0 + 3, y0 + 8),
            cv2.FONT_HERSHEY_PLAIN,
            font_scale,
            (0, 0, 255),
            thickness,
        )
        cv2.putText(
            img,
            f"col: {logic_point[2]}-{logic_point[3]}",
            (x0 + 3, y0 + 18),
            cv2.FONT_HERSHEY_PLAIN,
            font_scale,
            (0, 0, 255),
            thickness,
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # 保存绘制后的图像
        cv2.imwrite(output_path, img)


def plot_rec_box(img_path, output_path, sorted_polygons):
    """
    :param img_path
    :param output_path
    :param sorted_polygons: [xmin,ymin,xmax,ymax]
    :return:
    """
    # 处理ocr_res
    img = cv2.imread(img_path)
    img = cv2.copyMakeBorder(
        img, 0, 0, 0, 100, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    # 绘制 ocr_res 矩形
    for idx, polygon in enumerate(sorted_polygons):
        x0, y0, x1, y1 = polygon[0], polygon[1], polygon[2], polygon[3]
        x0 = round(x0)
        y0 = round(y0)
        x1 = round(x1)
        y1 = round(y1)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
        # 增大字体大小和线宽
        font_scale = 0.9  # 原先是0.5
        thickness = 1  # 原先是1

        cv2.putText(
            img,
            str(idx),
            (x0 + 5, y0 + 5),
            cv2.FONT_HERSHEY_PLAIN,
            font_scale,
            (0, 0, 255),
            thickness,
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 保存绘制后的图像
    cv2.imwrite(output_path, img)


def format_html(html):
    return f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
    <meta charset="UTF-8">
    <title>Complex Table Example</title>
    <style>
        table {{
            border-collapse: collapse;
            width: 100%;
        }}
        th, td {{
            border: 1px solid black;
            padding: 8px;
            text-align: center;
        }}
        th {{
            background-color: #f2f2f2;
        }}
    </style>
    </head>
    <body>
    {html}
    </body>
    </html>
    """
