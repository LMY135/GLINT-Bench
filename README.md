# GLINT-Bench

GLINT-Bench 是一个面向视频多模态大模型的**细粒度时空证据推理 Benchmark**。

Benchmark 主要评估模型是否能够：

1. 正确回答视频问题；
2. 定位支持答案的瞬时 Temporal Evidence；
3. 在正确时间进一步定位对应的 Spatial Evidence。

整体评测过程可以概括为：

```text
Answer Correctness
        ↓
Temporal Evidence Localization
        ↓
Spatial Evidence Localization
```

当前仓库用于统一管理 **数据来源、视频元数据、QA / Ground Truth、Annotation Pipeline、模型实验结果以及复现代码**。

## Setup
1. Clone repository
2. pip install -r requirements.txt
3. 安装 / 检查 FFmpeg、FFprobe
4. 创建本地 videos/
5. 把视频放入 videos/

---

## Repository Structure

```text
GLINT-Bench/
├── data/
│   ├── annotations/              # Local CVAT XML annotations
│   ├── sources.jsonl
│   ├── video_metadata.jsonl
│   └── qa.jsonl
│
├── videos/                       # Local benchmark videos
│
├── results/
│   └── README.md
│
├── scripts/
│   ├── annotation/
│   │   ├── README.md
│   │   ├── build_qa.py
│   │   ├── cvat.py
│   │   └── utils.py
│   │
│   ├── prepare_videos.py
│   └── extract_video_metadata.py
│
├── docs/
│   └── DATA_FORMAT.md
│
├── requirements.txt
├── README.md
└── LICENSE
```

其中：

```text
videos/
data/annotations/
```

主要作为本地数据准备和标注目录。

---

## `data/sources.jsonl`

记录 Benchmark 中所有视频的**原始来源**。

GLINT-Bench 的视频可以来自：

- **Existing Datasets**：已有的公开视频数据集；
- **Web Videos**：YouTube、Bilibili 等公开网络视频。

每条记录用于保存视频来源及 Benchmark Clip 与原视频之间的对应关系，例如：

```text
video_id
category
source_type
source_dataset
source_url
source_video_id
clip_start
clip_end
```

用于保证 Benchmark 中的视频来源可以追溯。

> 添加或查找视频来源 → `data/sources.jsonl`

---

## `data/video_metadata.jsonl`

记录已经准备好的 Benchmark Video 的基础元数据。

在使用该脚本前，需要先在项目根目录本地创建：

```text
videos/
```

并将已经准备好的 Benchmark Video 放入该目录，例如：

```text
GLINT-Bench/
├── videos/
│   ├── magic_0001.mp4
│   ├── magic_0002.mp4
│   └── game_0001.mp4
│
├── data/
├── scripts/
└── README.md
```

`extract_video_metadata.py` 默认从：

```text
videos/
```

读取本地视频，并根据视频文件名确定 `video_id`。

例如：

```text
videos/magic_0001.mp4
```

对应：

```text
video_id = magic_0001
```

提取后的元数据写入：

```text
data/video_metadata.jsonl
```

例如：

```json
{
  "video_id": "magic_0001",
  "width": 1080,
  "height": 1920,
  "fps": 30.0,
  "duration": 15.0,
  "num_frames": 450
}
```

主要字段包括：

```text
video_id
width
height
fps
duration
num_frames
```

这些信息用于后续：

```text
Frame Index
    ↓
Timestamp
```

以及检查 CVAT Annotation 与原视频分辨率是否一致。

视频放入本地 `videos/` 目录后，在项目根目录运行：

```bash
python scripts/extract_video_metadata.py
```

脚本会读取 `videos/` 中的视频，并生成或更新：

```text
data/video_metadata.jsonl
```

因此基本流程为：

```text
准备视频
    ↓
放入本地 videos/
    ↓
python scripts/extract_video_metadata.py
    ↓
data/video_metadata.jsonl
    ↓
后续 CVAT Annotation / QA 构建
```

> `videos/` 用于本地存放 Benchmark Video，原始第三方视频不直接随仓库重新分发。

---

## `data/qa.jsonl`

存放 GLINT-Bench 最终的 **QA 和 Spatio-Temporal Ground Truth**。

主要包括：

```text
question_id
video_id
category
question_type
question
answer
evidence_windows
evidence_boxes
```

其中：

- `question`：问题；
- `answer`：正确答案；
- `evidence_windows`：支持答案的 Temporal Evidence；
- `evidence_boxes`：对应 Evidence Frame 中的目标 Bounding Box。

Bounding Box 使用原视频像素坐标：

```text
[x, y, width, height]
```

当前 Question 主要使用：

```text
question_type = "mcq"
```

MCQ 的选项直接写入 `question`，`answer` 保存正确选项，例如：

```json
{
  "question_type": "mcq",
  "question": "What is the suit of the third card that appears in the person's hand? A. Hearts B. Diamonds C. Clubs D. Spades",
  "answer": "D"
}
```

> 写问题、查答案或检查 Ground Truth → `data/qa.jsonl`

---

## Annotation Pipeline

对于需要人工标注 Spatial Evidence 的视频，GLINT-Bench 使用 CVAT 进行 Bounding Box Track 标注。

整体流程为：

```text
Video
  ↓
CVAT Annotation
  ↓
CVAT XML
  ↓
scripts/annotation/cvat.py
  ↓
Track / Frame / Bounding Box
  ↓
scripts/annotation/build_qa.py
  +
video_metadata.jsonl
  ↓
data/qa.jsonl
```

CVAT 中：

```text
q01
q02
q03
...
```

分别对应不同 Question。

转换脚本会自动生成：

```text
question_id
video_id
category
evidence_windows
evidence_boxes
```

而：

```text
question_type
question
answer
```

由人工填写。

### 新增 Annotation

```bash
python scripts/annotation/build_qa.py
```

已有 Question 会跳过，新 Question 会添加到 `qa.jsonl`。

### 更新全部已有 Annotation

```bash
python scripts/annotation/build_qa.py --overwrite
```

### 更新指定 Question

```bash
python scripts/annotation/build_qa.py --overwrite magic_0001_q02
```

也可以一次更新多个：

```bash
python scripts/annotation/build_qa.py --overwrite magic_0001_q01 magic_0001_q02
```

Overwrite 时会重新生成自动字段，同时保留人工填写的：

```text
question_type
question
answer
```

更完整的 Annotation 使用说明见：

```text
scripts/annotation/README.md
```

---

## `docs/DATA_FORMAT.md`

包含 GLINT-Bench 数据格式的详细说明，包括：

```text
sources.jsonl
video_metadata.jsonl
qa.jsonl
results
```

以及各字段的定义和填写方式。

> 不知道某个字段怎么填 → `docs/DATA_FORMAT.md`

---

## `scripts/`

存放 Benchmark 的数据准备、Annotation 和后续复现代码。

当前主要包括：

```text
scripts/
├── annotation/
│   ├── build_qa.py
│   ├── cvat.py
│   └── utils.py
├── prepare_videos.py
└── extract_video_metadata.py
```

其中：

- `prepare_videos.py`：准备 Benchmark Clips；
- `extract_video_metadata.py`：提取视频 FPS、分辨率、时长和帧数；
- `annotation/`：将 CVAT Annotation 转换为最终 `qa.jsonl`。

---

## `results/`

存放不同 Video MLLM 在 GLINT-Bench 上的实验结果。

GLINT-Bench 当前主要从三个层面进行评估：

```text
Criterion I
Answer Correctness
→ AccQA

Criterion II
Instantaneous Temporal Grounding
→ TNWD

Criterion III
Instantaneous Spatio-Temporal Grounding
→ sIoU / TG-sIoU / CS-IoU
```

其中：

- **AccQA**：评估最终答案是否正确；
- **TNWD**：评估模型是否定位到正确的瞬时时间，并考虑预测时间段的中心偏移和持续时间；
- **sIoU**：评估 Temporal Intersection 内采样帧上的 Spatial Localization；
- **TG-sIoU**：使用 Temporal Grounding 对 Spatial IoU 进行约束；
- **CS-IoU**：评估 Temporal Grounding 有效时的条件 Spatial Localization 能力。

完整指标定义见：

```text
results/README.md
```

---

## Data Preparation

**为避免潜在的版权和数据许可问题，GLINT-Bench 不直接重新分发来自第三方 Dataset 或公开网络平台的原始视频。**

对于来自已有 Dataset 的视频，请根据：

```text
data/sources.jsonl
```

中记录的数据集来源获取原始数据。

对于 Web Video，请根据其中记录的原始 URL 获取对应视频。

所有第三方视频仍遵循其原始来源的版权、许可证及使用条款。

在用户自行获取原始视频后，可以通过数据准备脚本将视频整理或裁剪为 GLINT-Bench 所需的 Benchmark Clips。

本地视频统一放置于：

```text
videos/
```

随后可以生成：

```text
data/video_metadata.jsonl
```

并进行后续 Annotation 和 QA 构建。

---

## Benchmark Pipeline

完整的数据构建与评测流程为：

```text
Video Sources
     ↓
sources.jsonl
     ↓
Prepare Benchmark Videos
     ↓
videos/
     ↓
video_metadata.jsonl
     ↓
CVAT Annotation
     ↓
CVAT XML
     ↓
build_qa.py
     ↓
qa.jsonl
     ↓
Model Inference
     ↓
Answer Prediction
     ↓
Temporal Grounding
     ↓
Spatial Grounding
     ↓
Evaluation
     ↓
results/
```

GLINT-Bench 最终希望评估的不只是模型是否**答对问题**，还包括模型是否能够：

```text
答对问题
    ↓
找对时间
    ↓
找对视觉证据
```

即模型是否能够在**正确的时间，定位到支持答案的正确视觉证据**。
