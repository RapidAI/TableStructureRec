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


### 指标结果
|方法|TEDS|
|:---|:---|
|[RapidTable](https://github.com/RapidAI/RapidStructure/blob/b800b156015bf5cd6f5429295cdf48be682fd97e/docs/README_Table.md)|0.58786|
|[lineless_table_rec](https://rapidai.github.io/TableStructureRec/docs/install_usage/lineless_table_rec/)|0.50054|
|[wired_table_rec](https://rapidai.github.io/TableStructureRec/docs/install_usage/wired_table_rec/)|0.63316|


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

#### 4. 写在最后
以上评测仅是基于表格测试数据集[liekkas/table_recognition](https://www.modelscope.cn/datasets/liekkas/table_recognition/summary)测试而来，不能完全代表模型效果。

因为每个模型训练数据不同，测试数据集如与训练数据相差较大，难免效果较差，请针对自身场景客观看待评测指标。