---
weight: 3510
lastmod: "2023-11-21"
draft: false
author: "SWHL"
title: "lineless_table_rec"
icon: "table"
toc: true
description: ""‘
---

<p>
  <a href=""><img src="https://img.shields.io/badge/Python->=3.6,<3.12-aff.svg"></a>
  <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Mac%2C%20Win-pink.svg"></a>
  <a href="https://pepy.tech/project/lineless-table-rec"><img src="https://static.pepy.tech/badge/lineless-table-rec?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads"></a>
  <a href="https://pypi.org/project/lineless-table-rec/"><img alt="PyPI" src="https://img.shields.io/pypi/v/lineless-table-rec"></a>
</p>

### 简介
`lineless_table_rec`库源于[阿里读光-LORE无线表格结构识别模型](https://www.modelscope.cn/models/damo/cv_resnet-transformer_table-structure-recognition_lore/summary)。

在这里，我门做的工作主要包括以下两点：
1. 将模型转换为ONNX格式，便于部署
2. 完善后处理代码，与OCR识别模型整合，可以保证输出结果为完整的表格和对应的内容

### 安装
```bash {linenos=table}
pip install lineless_table_rec
```

### 使用
{{< tabs tabTotal="2">}}
{{% tab tabName="Python脚本使用" %}}


```python {linenos=table}
from lineless_table_rec import LinelessTableRecognition

engine = LinelessTableRecognition()

img_path = "tests/test_files/lineless_table_recognition.jpg"
table_str, elapse = engine(img_path)

print(table_str)
print(elapse)
```

{{% /tab %}}
{{% tab tabName="终端使用" %}}

```bash {lineos=table}
$ lineless_table_rec -img tests/test_files/lineless_table_recognition.jpg
```

{{% /tab %}}
{{< /tabs >}}


### 查看效果
<div align="center">
    <img src="https://github.com/RapidAI/TableStructureRec/releases/download/v0.0.0/lineless_table_rec_result.png">

</div>


<details>
    <summary>识别结果（点击展开）</summary>

```html {lineos=table}
<html>
<body>
    <table>
        <tbody>
            <tr>
                <td rowspan="1" colspan="1">姓名</td>
                <td rowspan="1" colspan="1">年龄</td>
                <td rowspan="1" colspan="1">性别</td>
                <td rowspan="1" colspan="1">身高/m</td>
                <td rowspan="1" colspan="1">体重/kg</td>
                <td rowspan="1" colspan="1">BMI/(kg/m²)</td>
            </tr>
            <tr>
                <td rowspan="1" colspan="1">Duke</td>
                <td rowspan="1" colspan="1">34</td>
                <td rowspan="1" colspan="1">男</td>
                <td rowspan="1" colspan="1">1.74</td>
                <td rowspan="1" colspan="1">70</td>
                <td rowspan="1" colspan="1">23</td>
            </tr>
            <tr>
                <td rowspan="1" colspan="1">Ella</td>
                <td rowspan="1" colspan="1">26</td>
                <td rowspan="1" colspan="1">女</td>
                <td rowspan="1" colspan="1">1.60</td>
                <td rowspan="1" colspan="1">58</td>
                <td rowspan="1" colspan="1">23</td>
            </tr>
            <tr>
                <td rowspan="1" colspan="1">Eartha</td>
                <td rowspan="1" colspan="1"></td>
                <td rowspan="1" colspan="1">女</td>
                <td rowspan="1" colspan="1">1.34</td>
                <td rowspan="1" colspan="1">29</td>
                <td rowspan="1" colspan="1">16</td>
            </tr>
            <tr>
                <td rowspan="1" colspan="1">Thelonious</td>
                <td rowspan="1" colspan="1">6</td>
                <td rowspan="1" colspan="1">男</td>
                <td rowspan="1" colspan="1">1.07</td>
                <td rowspan="1" colspan="1">17</td>
                <td rowspan="1" colspan="1">15</td>
            </tr>
            <tr>
                <td rowspan="1" colspan="1">TARO</td>
                <td rowspan="1" colspan="1">22</td>
                <td rowspan="1" colspan="1">男</td>
                <td rowspan="1" colspan="1">1.728</td>
                <td rowspan="1" colspan="1">65</td>
                <td rowspan="1" colspan="1">21.7</td>
            </tr>
            <tr>
                <td rowspan="1" colspan="1">HANAKO</td>
                <td rowspan="1" colspan="1">22</td>
                <td rowspan="1" colspan="1">女</td>
                <td rowspan="1" colspan="1">1.60</td>
                <td rowspan="1" colspan="1">53</td>
                <td rowspan="1" colspan="1">20.7</td>
            </tr>
            <tr>
                <td rowspan="1" colspan="1">NARMAN</td>
                <td rowspan="1" colspan="1">38</td>
                <td rowspan="1" colspan="1">男</td>
                <td rowspan="1" colspan="1">1.76</td>
                <td rowspan="1" colspan="1">73</td>
                <td rowspan="1" colspan="1"></td>
            </tr>
            <tr>
                <td rowspan="1" colspan="1">NAOMI</td>
                <td rowspan="1" colspan="1">23</td>
                <td rowspan="1" colspan="1">女</td>
                <td rowspan="1" colspan="1">1.63</td>
                <td rowspan="1" colspan="1">60</td>
                <td rowspan="1" colspan="1"></td>
            </tr>
        </tbody>
    </table>
</body>

</html>
```

</details>