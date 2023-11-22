---
weight: 3520
lastmod: "2023-11-22"
draft: false
author: "SWHL"
title: "wired_table_rec"
icon: "table"
toc: true
description: ""
---

<p>
  <a href=""><img src="https://img.shields.io/badge/Python->=3.6,<3.12-aff.svg"></a>
  <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Mac%2C%20Win-pink.svg"></a>
  <a href="https://pepy.tech/project/wired_table_rec"><img src="https://static.pepy.tech/badge/wired_table_rec?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads"></a>
  <a href="https://pypi.org/project/wired_table_rec/"><img alt="PyPI" src="https://img.shields.io/pypi/v/wired_table_rec"></a>
</p>


### 简介
`wired_table_rec`库源于[阿里读光-表格结构识别-有线表格](https://www.modelscope.cn/models/damo/cv_dla34_table-structure-recognition_cycle-centernet/summary)。

在这里，我们做的工作主要包括以下两点：
1. 将模型转换为ONNX格式，便于部署
2. 完善后处理代码，与OCR识别模型整合，可以保证输出结果为完整的表格和对应的内容

{{< alert context="info" text="该库仅提供推理代码，如有训练模型需求，需要参考modelscope中相关代码，该算法没有提供单独仓库。" />}}

### 安装
```bash {linenos=table}
pip install wired_table_rec
```

### 使用
{{< tabs tabTotal="2">}}
{{% tab tabName="Python脚本使用" %}}


```python {linenos=table}
from wired_table_rec import WiredTableRecognition


table_rec = WiredTableRecognition()

img_path = "tests/test_files/wired/table_recognition.jpg"
table_str, elapse = table_rec(img_path)
print(table_str)
print(elapse)
```

{{% /tab %}}
{{% tab tabName="终端使用" %}}

```bash {lineos=table}
$ wired_table_rec -img tests/test_files/wired/table_recognition.jpg
```

{{% /tab %}}
{{< /tabs >}}


### 查看效果
<div align="center">
    <img src="https://github.com/RapidAI/TableStructureRec/releases/download/v0.0.0/wired_table_rec_result.jpg">
</div>


<details>
    <summary>识别结果（点击展开）</summary>

```html {lineos=table}
<html>

<body>
    <table>
        <tr>
            <td rowspan="2">名称</td>
            <td rowspan="2">产量（吨）</td>
            <td colspan="2">环比</td>
        </tr>
        <tr>
            <td>增长量（吨）</td>
            <td>增长率（%)</td>
        </tr>
        <tr>
            <td>荔枝</td>
            <td>11</td>
            <td></td>
            <td>10</td>
        </tr>
        <tr>
            <td>芒果</td>
            <td></td>
            <td></td>
            <td>-10</td>
        </tr>
        <tr>
            <td>香蕉</td>
            <td></td>
            <td></td>
            <td>20</td>
        </tr>
    </table>
</body>

</html>
```

</details>