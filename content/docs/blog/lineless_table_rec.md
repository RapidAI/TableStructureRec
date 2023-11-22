---
weight: 3710
lastmod: "2023-11-21"
draft: false
author: "SWHL"
title: "LORE: 无线表格结构识别算法"
icon: "table"
toc: true
description: ""
---

### 引言
LORE算法来自论文[LORE: Logical Location Regression Network for Table Structure Recognition](https://arxiv.org/abs/2303.03730)，是阿里的一篇工作。

该工作主要解决无线表格结构识别问题，具体包括文档中涉及到一些三线表之类表格结构识别。对于有线的表格支持较差。


### 基本原理
主要原理为:

1）基于无线单元格中心点回归出到4个顶点的距离，解码出单元格bbox；

2）结合视觉特征与单元格bbox信息，采用两个级联回归器兼顾全局与局部注意力，直接对单元格的逻辑坐标进行回归；

3）模型训练时显式利用单元格间与单元格内逻辑约束对模型进行优化。

<div align="center">
    <img src="https://www.modelscope.cn/api/v1/models/damo/cv_resnet-transformer_table-structure-recognition_lore/repo?Revision=master&FilePath=./description/Pipeline.png&View=true">
</div>

### 参考资料
- [读光-表格结构识别-无线表格](https://www.modelscope.cn/models/damo/cv_resnet-transformer_table-structure-recognition_lore/summary)