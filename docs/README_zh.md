<div align="center">
  <div align="center">
    <h1><b>📊 表格结构识别</b></h1>
  </div>
  <a href=""><img src="https://img.shields.io/badge/Python->=3.6,<3.12-aff.svg"></a>
  <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Mac%2C%20Win-pink.svg"></a>
  <a href="https://semver.org/"><img alt="SemVer2.0" src="https://img.shields.io/badge/SemVer-2.0-brightgreen"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
  <a href="https://github.com/RapidAI/TableStructureRec/blob/c41bbd23898cb27a957ed962b0ffee3c74dfeff1/LICENSE"><img alt="GitHub" src="https://img.shields.io/badge/license-Apache 2.0-blue"></a>

  简体中文 | [English](https://github.com/RapidAI/TableStructureRec)
</div>

### 简介
该仓库是用来对文档中表格做结构化识别的推理库，包括来自PaddleOCR的表格结构识别算法模型、来自阿里读光有线和无线表格识别算法模型等。

该仓库将表格识别前后处理做了完善，并结合OCR，保证表格识别部分可直接使用。

### 表格结构化识别
表格结构识别（Table Structure Recognition, TSR）旨在提取表格图像的逻辑或物理结构，从而将非结构化的表格图像转换为机器可读的格式。

逻辑结构：表示单元格的行/列关系（例如同行、同列）和单元格的跨度信息。

物理结构：不仅包含逻辑结构，还包含单元格的包围框、内容等信息，强调单元格的物理位置。

<div align='center'>
   <img src="https://github.com/RapidAI/TableStructureRec/releases/download/v0.0.0/TSRFramework.jpg" width=70%>
</div>

图来自： [Improving Table Structure Recognition with Visual-Alignment Sequential Coordinate Modeling](https://openaccess.thecvf.com/content/CVPR2023/html/Huang_Improving_Table_Structure_Recognition_With_Visual-Alignment_Sequential_Coordinate_Modeling_CVPR_2023_paper.html)

### 文档
完整文档请移步：[docs](https://rapidai.github.io/TableStructureRec/docs/)

### 致谢
[PaddleOCR 表格识别](https://github.com/PaddlePaddle/PaddleOCR/blob/4b17511491adcfd0f3e2970895d06814d1ce56cc/ppstructure/table/README_ch.md)

[读光-表格结构识别-有线表格](https://www.modelscope.cn/models/damo/cv_dla34_table-structure-recognition_cycle-centernet/summary)

[读光-表格结构识别-无线表格](https://www.modelscope.cn/models/damo/cv_resnet-transformer_table-structure-recognition_lore/summary)


### 贡献指南
欢迎提交请求。对于重大更改，请先打开issue讨论您想要改变的内容。

请确保适当更新测试。

### [赞助](https://rapidai.github.io/Knowledge-QA-LLM/docs/sponsor/)
如果您想要赞助该项目，可直接点击当前页最上面的Sponsor按钮，请写好备注(**您的Github账号名称**)，方便添加到赞助列表中。


### 开源许可证
该项目采用[Apache 2.0](https://github.com/RapidAI/TableStructureRec/blob/c41bbd23898cb27a957ed962b0ffee3c74dfeff1/LICENSE)开源许可证。