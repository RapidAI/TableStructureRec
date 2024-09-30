---
weight: 3730
lastmod: "2023-11-30"
draft: false
author: "SWHL"
title: "三个表格识别算法评测"
icon: "table"
toc: true
description: ""
---

### 引言
为了便于比较不同表格识别算法的效果差异，本篇文章基于评测工具[TableRecognitionMetric](https://github.com/SWHL/TableRecognitionMetric)和表格测试数据集[liekkas/table_recognition](https://www.modelscope.cn/datasets/liekkas/table_recognition/summary)上计算不同算法的TEDS指标。

以下评测仅是基于表格测试数据集[liekkas/table_recognition](https://www.modelscope.cn/datasets/liekkas/table_recognition/summary)测试而来，不能完全代表模型效果。

因为每个模型训练数据不同，测试数据集如与训练数据相差较大，难免效果较差，请针对自身场景客观看待评测指标。

**RapidTable**: 有英文和中文两个模型，大多都是印刷体截屏表格。具体可参见:[表格识别模型](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/ppstructure/docs/models_list.md#22-%E8%A1%A8%E6%A0%BC%E8%AF%86%E5%88%AB%E6%A8%A1%E5%9E%8B)。

**lineless_table_rec**: 训练数据部分来自SciTSR与PubTabNet，训练集共45000张。这两个数据大多是来自论文截图，所以这个模型也是比较适用于论文中表格识别。

**wired_table_rec**: 训练数据为WTW，训练集为10970张。WTW数据组成有50%的自然场景下、30%的档案和20%的印刷体表格。所以这个模型更适合自然场景下拍照的表格识别。

### 指标结果
| 方法                                                                                                                        |    TEDS     | TEDS-only-structure |
|:---------------------------------------------------------------------------------------------------------------------------|:-----------:|:-------------------:|
| [deepdoctection(rag-flow)](https://github.com/deepdoctection/deepdoctection?tab=readme-ov-file)                            |   0.59975   |       0.69918       |
| [ppstructure_table_master](https://github.com/PaddlePaddle/PaddleOCR/tree/main/ppstructure)                                |   0.61606   |       0.73892       |
| [ppsturcture_table_engine](https://github.com/PaddlePaddle/PaddleOCR/tree/main/ppstructure)                                |   0.67924   |       0.78653       |
| table_cls + wired_table_rec v1 + lineless_table_rec                                                                        |   0.68507   |       0.75140       |
| [StructEqTable](https://github.com/UniModal4Reasoning/StructEqTable-Deploy)                                                |   0.67310   |     **0.81210**     |
| [RapidTable](https://github.com/RapidAI/RapidStructure/blob/b800b156015bf5cd6f5429295cdf48be682fd97e/docs/README_Table.md) |   0.71654   |       0.81067       |
| table_cls + wired_table_rec v2 + lineless_table_rec                                                                        | **0.73702** |       0.80210       |


### 评测步骤
#### 1. 安装评测数据集和评测工具包
```bash {linenos=table}
pip install table_recognition_metric
pip install modelscope==1.5.2
pip install rapidocr_onnxruntime==1.3.8
```

#### 2. 安装表格识别推理库
```bash {linenos=table}
pip install rapid_table
pip install lineless_table_rec
pip install wired_table_rec
```

#### 3. 推理代码
{{< alert context="info" text="完整评测代码，请移步[Gist](https://gist.github.com/SWHL/4218b337f37ae07acd6ba859bae39d33)" />}}

```python {linenos=table}
from modelscope.msdatasets import MsDataset

from rapid_table import RapidTable
from lineless_table_rec import LinelessTableRecognition
from wired_table_rec import WiredTableRecognition

from table_recognition_metric import TEDS

test_data = MsDataset.load(
    "table_recognition",
    namespace="liekkas",
    subset_name="default",
    split="test",
)

# 这里依次更换不同算法实例即可
table_engine = RapidTable()
# table_engine = LinelessTableRecognition()
# table_engine = WiredTableRecognition()
teds = TEDS()

content = []
for one_data in test_data:
    img_path = one_data.get("image:FILE")
    gt = one_data.get("label")

    pred_str, _ = table_engine(img_path)
    scores = teds(gt, pred_str)
    content.append(scores)
    print(f"{img_path}\t{scores:.5f}")

avg = sum(content) / len(content)
print(f'{avg:.5f}')
```
