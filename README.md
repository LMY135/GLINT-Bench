# GLINT-Bench

GLINT-Bench 是一个面向视频多模态大模型的细粒度时空证据推理 Benchmark。

当前仓库用于统一管理 **数据来源、QA / Ground Truth、模型实验结果以及复现代码**。

## Repository Structure

```text
GLINT-Bench/
├── data/
│   ├── sources.jsonl
│   └── qa.jsonl
│
├── results/
├── scripts/
│
├── docs/
│   └── DATA_FORMAT.md
│
├── README.md
└── LICENSE
```

### `data/sources.jsonl`

记录 Benchmark 中所有视频的**原始来源**。

GLINT-Bench 的视频来自两类来源：

* **Existing Datasets**：已有的公开视频数据集；
* **Web Videos**：YouTube、Bilibili 等公开网络视频。

这里记录每个视频对应的原 Dataset / URL、原视频 ID、使用片段等信息，保证数据来源可以追溯。

> 添加或查找视频来源 → 看 `sources.jsonl`

### `data/qa.jsonl`

存放 Benchmark 的 **QA 和 Ground Truth**，包括：

* Question / Answer
* Temporal Evidence
* Spatial Evidence
* Task / Reasoning Type

> 写问题、查答案或做 Annotation → 看 `qa.jsonl`

### `docs/DATA_FORMAT.md`

详细的数据格式说明，包括 `sources.jsonl`、`qa.jsonl` 和实验结果各字段的定义与填写方式。

> 不知道某个字段怎么填 → 先看 `DATA_FORMAT.md`

### `scripts/`

存放 Benchmark 的**复现代码**，包括后续的视频准备、模型推理和 Evaluation 脚本。

### `results/`

存放不同模型在 GLINT-Bench 上的实验结果，包括 Answer、Temporal 和 Spatial Prediction。

## Data Preparation

**为避免潜在的版权和数据许可问题，GLINT-Bench 不直接重新分发来自第三方 Dataset 或公开网络平台的原始视频。**

对于来自已有 Dataset 的视频，请根据 `data/sources.jsonl` 中记录的 Dataset 来源获取原始数据；对于 Web 视频，请根据其中记录的原始 URL 获取对应视频。所有第三方视频仍遵循其原始来源的版权、许可证及使用条款。

后续将提供统一的数据准备脚本，在用户自行获取原始数据后，将其整理和裁剪为 GLINT-Bench 所需的 Benchmark Clips。
