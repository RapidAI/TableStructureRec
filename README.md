<div align="center">
  <div align="center">
    <h1><b>📊 Table Structure Recognition</b></h1>
  </div>
  <a href=""><img src="https://img.shields.io/badge/Python->=3.6,<3.12-aff.svg"></a>
  <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Mac%2C%20Win-pink.svg"></a>
<a href="https://pypi.org/project/lineless-table-rec/"><img alt="PyPI" src="https://img.shields.io/pypi/v/lineless-table-rec"></a>
<a href="https://pepy.tech/project/lineless-table-rec"><img src="https://static.pepy.tech/personalized-badge/lineless-table-rec?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads%20Lineless"></a>
<a href="https://pepy.tech/project/wired-table-rec"><img src="https://static.pepy.tech/personalized-badge/wired-table-rec?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads%20Wired"></a>
  <a href="https://semver.org/"><img alt="SemVer2.0" src="https://img.shields.io/badge/SemVer-2.0-brightgreen"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
  <a href="https://github.com/RapidAI/TableStructureRec/blob/c41bbd23898cb27a957ed962b0ffee3c74dfeff1/LICENSE"><img alt="GitHub" src="https://img.shields.io/badge/license-Apache 2.0-blue"></a>

  [简体中文](./docs/README_zh.md) | English
</div>

### Introduction

This repository is a library for structured recognition of tables in documents. 
It includes table recognition models from Paddle, Alibaba's DocLight wired and wireless table recognition models, 
wired table models contributed by others, and the built-in table classification model from NetEase QAnything.



#### Features
⚡  **Fast**: Uses ONNXRuntime as the inference engine, achieving 1-7 second inference times on CPU.

🎯 **Accurate**: Combines table type classification models to distinguish between wired and wireless tables, leading to more specialized tasks and higher accuracy.

🛡️ **Stable**: Does not depend on any third-party training frameworks, uses specialized ONNX models, and completely solves memory leak issues.

### Results Demonstration
<div align="center">
    <img src="https://github.com/RapidAI/TableStructureRec/releases/download/v0.0.0/demo_img_output.gif" alt="Demo" width="100%" height="100%">
</div>

### 指标结果
[TableRecognitionMetric](https://github.com/SWHL/TableRecognitionMetric)

[dataset](https://huggingface.co/datasets/SWHL/table_rec_test_dataset)

[Rapid OCR](https://github.com/RapidAI/RapidOCR)

| model                                                                                                                      |TEDS|
|:---------------------------------------------------------------------------------------------------------------------------|:-|
| lineless_table_rec                                                                                                         |0.50054|
| [RapidTable](https://github.com/RapidAI/RapidStructure/blob/b800b156015bf5cd6f5429295cdf48be682fd97e/docs/README_Table.md) |0.58786|
| wired_table_rec v1                                                                                                         |0.70279|
| table_cls + wired_table_rec v1 + lineless_table_rec                                                                        |0.74692|
| table_cls + wired_table_rec v2 + lineless_table_rec                                                                        |0.80235|

### Install
``` python {linenos=table}
pip install wired_table_rec lineless_table_rec table_cls
```

### Quick Start
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
- [ ] rotate img fix before rec
- [ ] Increase dataset size
- [ ] Lineless table rec optimization
- 
### Acknowledgements

[PaddleOCR Table](https://github.com/PaddlePaddle/PaddleOCR/blob/4b17511491adcfd0f3e2970895d06814d1ce56cc/ppstructure/table/README_ch.md)

[Cycle CenterNet](https://www.modelscope.cn/models/damo/cv_dla34_table-structure-recognition_cycle-centernet/summary)

[LORE](https://www.modelscope.cn/models/damo/cv_resnet-transformer_table-structure-recognition_lore/summary)

[Qanything-RAG](https://github.com/netease-youdao/QAnything)

llaipython (WeChat, commercial support for table extraction) provides high-precision wired table models.

### Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

### [Sponsor](https://rapidai.github.io/Knowledge-QA-LLM/docs/sponsor/)

If you want to sponsor the project, you can directly click the **Buy me a coffee** image, please write a note (e.g. your github account name) to facilitate adding to the sponsorship list below.

<div align="left">
   <a href="https://www.buymeacoffee.com/SWHL"><img src="https://raw.githubusercontent.com/RapidAI/.github/main/assets/buymeacoffe.png" width="30%" height="30%"></a>
</div>

### License

This project is released under the [Apache 2.0 license](https://github.com/RapidAI/TableStructureRec/blob/c41bbd23898cb27a957ed962b0ffee3c74dfeff1/LICENSE).
