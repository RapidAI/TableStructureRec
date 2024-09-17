<div align="center">
  <div align="center">
    <h1><b>📊 表格结构识别</b></h1>
  </div>
  <a href=""><img src="https://img.shields.io/badge/Python->=3.6,<3.12-aff.svg"></a>
  <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Mac%2C%20Win-pink.svg"></a>
<a href="https://pypi.org/project/lineless-table-rec/"><img alt="PyPI" src="https://img.shields.io/pypi/v/lineless-table-rec"></a>
<a href="https://pepy.tech/project/lineless-table-rec"><img src="https://static.pepy.tech/personalized-badge/lineless-table-rec?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads%20Lineless"></a>
<a href="https://pepy.tech/project/wired-table-rec"><img src="https://static.pepy.tech/personalized-badge/wired-table-rec?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads%20Wired"></a>
  <a href="https://semver.org/"><img alt="SemVer2.0" src="https://img.shields.io/badge/SemVer-2.0-brightgreen"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
  <a href="https://github.com/RapidAI/TableStructureRec/blob/c41bbd23898cb27a957ed962b0ffee3c74dfeff1/LICENSE"><img alt="GitHub" src="https://img.shields.io/badge/license-Apache 2.0-blue"></a>

  简体中文 | [English](https://github.com/RapidAI/TableStructureRec)
</div>

### 简介

💖该仓库是用来对文档中表格做结构化识别的推理库，包括来自paddle的表格识别模型，
阿里读光有线和无线表格识别模型，其他人贡献的有线表格模型，网易Qanything内置表格分类模型等。

#### 特点
⚡  **快**  采用ONNXRuntime作为推理引擎，cpu下单图推理1-7s

🎯 **准**: 结合表格类型分类模型，区分有线表格，无线表格，任务更细分，精度更高

🛡️ **稳**: 不依赖任何第三方训练框架，采用onnx专项小模型, 彻底解决了内存泄露问题


### 效果展示
<div align="center">
    <img src="https://github.com/RapidAI/TableStructureRec/releases/download/v0.0.0/demo_img_output.gif" alt="Demo" width="100%" height="100%">
</div>

### 指标结果
[TableRecognitionMetric 评测工具](https://github.com/SWHL/TableRecognitionMetric)

[评测数据集](https://huggingface.co/datasets/SWHL/table_rec_test_dataset)

[Rapid OCR](https://github.com/RapidAI/RapidOCR)

| 方法                                                                                                                         |TEDS|
|:---------------------------------------------------------------------------------------------------------------------------|:-|
| lineless_table_rec                                                                                                         |0.53561|
| [RapidTable](https://github.com/RapidAI/RapidStructure/blob/b800b156015bf5cd6f5429295cdf48be682fd97e/docs/README_Table.md) |0.58786|
| wired_table_rec v1                                                                                                         |0.70279|
| wired_table_rec v2                                                                                                         |0.78007|
| table_cls + wired_table_rec v1 + lineless_table_rec                                                                        |0.74692|
| table_cls + wired_table_rec v2 + lineless_table_rec                                                                        |0.80235|

### 安装
``` python {linenos=table}
pip install wired_table_rec lineless_table_rec table_cls
```

### 快速使用
``` python {linenos=table}
import os

from lineless_table_rec import LinelessTableRecognition
from lineless_table_rec.utils_table_recover import format_html, plot_rec_box_with_logic_info, plot_rec_box
from table_cls import TableCls
from wired_table_rec import WiredTableRecognition

lineless_engine = LinelessTableRecognition()
wired_engine = WiredTableRecognition()
table_cls = TableCls()
img_path = f'images/img14.jpg'

cls,elasp = table_cls(img_path)
if cls == 'wired':
    table_engine = wired_engine
else:
    table_engine = lineless_engine
html, elasp, polygons, logic_points, ocr_res = table_engine(img_path)
print(f"elasp: {elasp}")

# output_dir = f'outputs'
# complete_html = format_html(html)
# os.makedirs(os.path.dirname(f"{output_dir}/table.html"), exist_ok=True)
# with open(f"{output_dir}/table.html", "w", encoding="utf-8") as file:
#     file.write(complete_html)
# # 可视化表格识别框 + 逻辑行列信息
# plot_rec_box_with_logic_info(
#     img_path, f"{output_dir}/table_rec_box.jpg", logic_points, polygons
# )
# # 可视化 ocr 识别框
# plot_rec_box(img_path, f"{output_dir}/ocr_box.jpg", ocr_res)
```
### TODO List 
- [ ] 识别前图片偏移修正
- [ ] 增加数据集数量，增加更多评测对比
- [ ] 优化无线表格模型

### 致谢

[PaddleOCR 表格识别](https://github.com/PaddlePaddle/PaddleOCR/blob/4b17511491adcfd0f3e2970895d06814d1ce56cc/ppstructure/table/README_ch.md)

[读光-表格结构识别-有线表格](https://www.modelscope.cn/models/damo/cv_dla34_table-structure-recognition_cycle-centernet/summary)

[读光-表格结构识别-无线表格](https://www.modelscope.cn/models/damo/cv_resnet-transformer_table-structure-recognition_lore/summary)

[Qanything-RAG](https://github.com/netease-youdao/QAnything)

llaipython(微信，商业化支持表格提取) 提供高精度有线表格模型。

### 贡献指南

欢迎提交请求。对于重大更改，请先打开issue讨论您想要改变的内容。

请确保适当更新测试。

### [赞助](https://rapidai.github.io/Knowledge-QA-LLM/docs/sponsor/)

如果您想要赞助该项目，可直接点击当前页最上面的Sponsor按钮，请写好备注(**您的Github账号名称**)，方便添加到赞助列表中。

### 开源许可证

该项目采用[Apache 2.0](https://github.com/RapidAI/TableStructureRec/blob/c41bbd23898cb27a957ed962b0ffee3c74dfeff1/LICENSE)开源许可证。
