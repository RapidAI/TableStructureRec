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
</div>

### 最近更新
- **2024.9.26**
  - 修正RapidTable默认英文模型导致的测评结果错误。
  - 补充测评数据集，补充开源社区更多模型的测评结果

### 简介
💖该仓库是用来对文档中表格做结构化识别的推理库，包括来自paddle的表格识别模型，
阿里读光有线和无线表格识别模型，llaipython(微信)贡献的有线表格模型，网易Qanything内置表格分类模型等。

#### 特点

⚡  **快**  采用ONNXRuntime作为推理引擎，cpu下单图推理1-7s

🎯 **准**: 结合表格类型分类模型，区分有线表格，无线表格，任务更细分，精度更高

🛡️ **稳**: 不依赖任何第三方训练框架，只依赖必要基础库，避免包冲突

### 效果展示

<div align="center">
    <img src="https://github.com/RapidAI/TableStructureRec/releases/download/v0.0.0/demo_img_output.gif" alt="Demo" width="100%" height="100%">
</div>

### 指标结果

[TableRecognitionMetric 评测工具](https://github.com/SWHL/TableRecognitionMetric) [huggingface数据集](https://huggingface.co/datasets/SWHL/table_rec_test_dataset) [modelscope 数据集](https://www.modelscope.cn/datasets/jockerK/TEDS_TEST/files) [Rapid OCR](https://github.com/RapidAI/RapidOCR)

注: StructEqTable 输出为 latex，只取成功转换为html并去除样式标签后进行测评

| 方法                                                                                                                        |    TEDS     | TEDS-only-structure |
|:---------------------------------------------------------------------------------------------------------------------------|:-----------:|:-------------------:|
| [deepdoctection(rag-flow)](https://github.com/deepdoctection/deepdoctection?tab=readme-ov-file)                            |   0.59975   |       0.69918       |
| [ppstructure_table_master](https://github.com/PaddlePaddle/PaddleOCR/tree/main/ppstructure)                                |   0.61606   |       0.73892       |
| [ppsturcture_table_engine](https://github.com/PaddlePaddle/PaddleOCR/tree/main/ppstructure)                                |   0.67924   |       0.78653       |
| table_cls + wired_table_rec v1 + lineless_table_rec                                                                        |   0.68507   |       0.75140       |
| [StructEqTable](https://github.com/UniModal4Reasoning/StructEqTable-Deploy)                                                |   0.67310   |     **0.81210**     |
| [RapidTable](https://github.com/RapidAI/RapidStructure/blob/b800b156015bf5cd6f5429295cdf48be682fd97e/docs/README_Table.md) |   0.71654   |       0.81067       |
| table_cls + wired_table_rec v2 + lineless_table_rec                                                                        | **0.73702** |       0.80210       |


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

# 使用其他ocr模型
#ocr_engine =RapidOCR(det_model_dir="xxx/det_server_infer.onnx",rec_model_dir="xxx/rec_server_infer.onnx")
#ocr_res, _ = ocr_engine(img_path)
#html, elasp, polygons, logic_points, ocr_res = table_engine(img_path, ocr_result=ocr_res)  

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

#### 偏移修正

```python
import cv2

img_path = f'tests/test_files/wired/squeeze_error.jpeg'
from wired_table_rec.utils import ImageOrientationCorrector

img_orientation_corrector = ImageOrientationCorrector()
img = cv2.imread(img_path)
img = img_orientation_corrector(img)
cv2.imwrite(f'img_rotated.jpg', img)
```

## FAQ (Frequently Asked Questions)

1. **问：偏移的图片能够处理吗？**
    - 答：该项目暂时不支持偏移图片识别，请先修正图片，也欢迎提pr来解决这个问题。

2. **问：识别框丢失了内部文字信息**
   - 答：默认使用的rapidocr小模型，如果需要更高精度的效果，可以从 [模型列表](https://rapidai.github.io/RapidOCRDocs/model_list/#_1)
   下载更高精度的ocr模型,在执行时传入ocr_result即可

3. **问：模型支持 gpu 加速吗？**
    - 答：目前表格模型的推理非常快，有线表格在100ms级别，无线表格在500ms级别，
      主要耗时在ocr阶段，可以参考 [rapidocr_paddle](https://rapidai.github.io/RapidOCRDocs/install_usage/rapidocr_paddle/usage/#_3)
      加速ocr识别过程

### TODO List

- [x] 图片小角度偏移修正方法补充
- [x] 增加数据集数量，增加更多评测对比
- [ ] 补充复杂场景表格检测和提取，解决旋转和透视导致的低识别率
- [ ] 优化无线表格模型

### 处理流程

```mermaid
flowchart TD
    A[/表格图片/] --> B([表格分类 table_cls])
    B --> C([有线表格识别 wired_table_rec]) & D([无线表格识别 lineless_table_rec]) --> E([文字识别 rapidocr_onnxruntime])
    E --> F[/html结构化输出/]
```

### 致谢

[PaddleOCR 表格识别](https://github.com/PaddlePaddle/PaddleOCR/blob/4b17511491adcfd0f3e2970895d06814d1ce56cc/ppstructure/table/README_ch.md)

[读光-表格结构识别-有线表格](https://www.modelscope.cn/models/damo/cv_dla34_table-structure-recognition_cycle-centernet/summary)

[读光-表格结构识别-无线表格](https://www.modelscope.cn/models/damo/cv_resnet-transformer_table-structure-recognition_lore/summary)

[Qanything-RAG](https://github.com/netease-youdao/QAnything)

非常感谢 llaipython(微信，提供全套有偿高精度表格提取) 提供高精度有线表格模型。

### 贡献指南

欢迎提交请求。对于重大更改，请先打开issue讨论您想要改变的内容。

请确保适当更新测试。

### [赞助](https://rapidai.github.io/Knowledge-QA-LLM/docs/sponsor/)

如果您想要赞助该项目，可直接点击当前页最上面的Sponsor按钮，请写好备注(**您的Github账号名称**)，方便添加到赞助列表中。

### 开源许可证

该项目采用[Apache 2.0](https://github.com/RapidAI/TableStructureRec/blob/c41bbd23898cb27a957ed962b0ffee3c74dfeff1/LICENSE)
开源许可证。
