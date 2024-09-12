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

This repo is an inference library used for structured recognition of tables in documents, including table structure recognition algorithm models from PaddleOCR, wired and wireless table recognition algorithm models from Alibaba Duguang, etc.

The repo has improved the pre- and post-processing of form recognition and combined with OCR to ensure that the form recognition part can be used directly.

The repo will continue to focus on the field of table recognition, integrate the latest and most useful table recognition algorithms, and strive to create the most valuable table recognition tool library.

Welcome everyone to continue to pay attention.

### What is Table Structure Recognition?

Table Structure Recognition (TSR) aims to extract the logical or physical structure of table images, thereby converting unstructured table images into machine-readable formats.

Logical structure: represents the row/column relationship of cells (such as the same row, the same column) and the span information of cells.

Physical structure: includes not only the logical structure, but also the cell's bounding box, content and other information, emphasizing the physical location of the cell.

<div align='center'>
   <img src="https://github.com/RapidAI/TableStructureRec/releases/download/v0.0.0/TSRFramework.jpg" width=70%>
</div>

Figure from: [Improving Table Structure Recognition with Visual-Alignment Sequential Coordinate Modeling](https://openaccess.thecvf.com/content/CVPR2023/html/Huang_Improving_Table_Structure_Recognition_With_Visual-Alignment_Sequential_Coordinate_Modeling_CVPR_2023_paper.html)

### Documentation

Full documentation can be found on [docs](https://rapidai.github.io/TableStructureRec/docs/), in Chinese.

### Acknowledgements

[PaddleOCR Table](https://github.com/PaddlePaddle/PaddleOCR/blob/4b17511491adcfd0f3e2970895d06814d1ce56cc/ppstructure/table/README_ch.md)

[Cycle CenterNet](https://www.modelscope.cn/models/damo/cv_dla34_table-structure-recognition_cycle-centernet/summary)

[LORE](https://www.modelscope.cn/models/damo/cv_resnet-transformer_table-structure-recognition_lore/summary)

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
