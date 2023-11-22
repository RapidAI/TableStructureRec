---
weight: 3710
lastmod: "2023-11-21"
draft: false
author: "SWHL"
title: "Cycle-CenterNet: 无线表格结构识别算法"
icon: "table"
toc: true
description: ""
---

### 引言
Cycle-CenterNet算法来自论文[Parsing Table Structure in the Wild](https://arxiv.org/abs/2109.02199)，是阿里的一篇工作。

该工作主要解决拍照和截屏场景下有线结构识别问题。


### 基本原理
本模型是以自底向上的方式:

1）基于单元格中心点回归出到4个顶点的距离，解码出单元格bbox；同时基于单元格顶点，回归出到共用该顶点的单元格的中心点距离，解码出gbox。

2）基于gbox(group box)，将离散的bbox拼接起来得到精准完整的电子表格；

3）第二步的拼接将单元格从“离散”变为“连续”，因此用后处理算法获得单元格的行列信息。

<div align="center">
    <img src="https://github.com/wangwen-whu/WTW-Dataset/blob/7a9c00f7d22a10d37d27b812608839c97596d966/demo/20210816_210413.gif">
</div>

### 参考资料
- [读光-表格结构识别-有线表格](https://www.modelscope.cn/models/damo/cv_dla34_table-structure-recognition_cycle-centernet/summary)