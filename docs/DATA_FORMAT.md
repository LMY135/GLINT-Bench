# GLINT-Bench 数据格式说明

本文档说明 GLINT-Bench 当前使用的数据格式。目前数据目录主要包含：

```text
data/
├── sources.jsonl
├── video_metadata.jsonl
└── qa.jsonl
```

* `sources.jsonl`：记录每条视频的来源以及从原视频中截取的时间范围。
* `video_metadata.jsonl`：记录裁剪后 benchmark 视频的分辨率、帧率、时长和帧数等信息。
* `qa.jsonl`：记录问题、答案及对应的时空证据 GT。**当前格式尚未更新，将在后续标注方案确定后补充。**

数据文件均采用 **JSONL（JSON Lines）** 格式，即每一行是一条独立的 JSON 记录。

---

## 1. `sources.jsonl`

`sources.jsonl` 用于记录每个 benchmark 视频片段的来源信息。

示例：

```json
{"video_id":"game_0023","category":"game","source_type":"web","source_dataset":null,"source_url":"https://www.youtube.com/shorts/xEaXEM9rI5I","source_video_id":"xEaXEM9rI5I","clip_start":2.0,"clip_end":20.0}
```

### 字段说明

| 字段                | 说明                                                                 |
| ----------------- | ------------------------------------------------------------------ |
| `video_id`        | GLINT-Bench 中该视频片段的唯一 ID。命名格式为 `{category}_{四位编号}`，例如 `game_0023`。 |
| `category`        | 视频所属类别。目前包括 `sports`、`traffic`、`crowd`、`game`、`magic`。             |
| `source_type`     | 视频来源类型。目前使用 `web` 或 `dataset`。                                     |
| `source_dataset`  | 如果视频来自已有数据集，填写数据集名称；如果来自 YouTube、Bilibili 等公开视频，则填写 `null`。        |
| `source_url`      | 原始视频或原始数据集的来源链接。                                                   |
| `source_video_id` | 原始来源中的视频 ID 或 sequence ID。例如 YouTube Video ID、Bilibili BV 号等。      |
| `clip_start`      | benchmark 片段在原始视频中的开始时间，单位为秒。                                      |
| `clip_end`        | benchmark 片段在原始视频中的结束时间，单位为秒。                                      |

### 时间说明

`sources.jsonl` 中的 `clip_start` 和 `clip_end` 都是相对于**原始来源视频**的时间。

例如：

```json
"clip_start": 120.0,
"clip_end": 133.0
```

表示从原始视频的 **2:00 到 2:13** 截取该 benchmark 视频。

---

## 2. `video_metadata.jsonl`

`video_metadata.jsonl` 用于记录**最终裁剪完成后的 benchmark 视频**的技术信息。

这些信息通过 `ffprobe` 从本地视频中自动读取，不需要人工填写。

示例：

```json
{"video_id":"game_0023","width":1080,"height":1920,"fps":30.0,"duration":18.0,"num_frames":540}
```

### 字段说明

| 字段           | 说明                                |
| ------------ | --------------------------------- |
| `video_id`   | 对应 `sources.jsonl` 中的 `video_id`。 |
| `width`      | 视频帧宽度，单位为像素（pixel）。               |
| `height`     | 视频帧高度，单位为像素（pixel）。               |
| `fps`        | 视频帧率（Frames Per Second）。          |
| `duration`   | 最终 benchmark 视频片段的时长，单位为秒。        |
| `num_frames` | 视频总帧数。                            |

不同视频**不要求具有相同的分辨率或 FPS**。每条视频分别保存自己的 metadata。

---

## 3. 视频坐标系与空间证据标注

GLINT-Bench 的空间证据统一使用**原始视频帧的像素坐标系**。坐标原点位于视频帧的**左上角 `(0, 0)`**，无需对坐标进行归一化。

Bounding Box 统一表示为：

```text
[x, y, width, height]
```

其中：

- `x`：Bounding Box 左上角的横坐标；
- `y`：Bounding Box 左上角的纵坐标；
- `width`：Bounding Box 的宽度；
- `height`：Bounding Box 的高度。

所有数值均为**原视频的绝对像素坐标**。

例如：

```json
"box": [412, 286, 173, 241]
```

表示 Bounding Box 左上角为 `(412, 286)`，宽度为 `173 px`，高度为 `241 px`。

每个视频的 `width`、`height`、`fps`、`duration` 和 `num_frames` 记录在：

```text
data/video_metadata.jsonl
```

因此，可以通过 `video_id` 查询对应视频的元数据，确定 Bounding Box 所使用的原视频坐标空间，并建立帧号与时间戳之间的对应关系。

### 空间证据来源

目前 `game` 和 `magic` 类视频主要使用 **CVAT** 在原始视频上进行人工空间证据标注。

CVAT 中标注的 Rectangle Track 会在转换为 GLINT-Bench QA 数据时统一转换为：

```text
[x, y, width, height]
```

对于 `sports`、`traffic` 和 `crowd` 等已有数据集，如果原数据集已经提供 Bounding Box 或 Track Ground Truth，则直接使用原数据集标注，并转换为 GLINT-Bench 的统一格式，无需重新使用 CVAT 标注。

因此目前存在两种标注来源：

```text
Game / Magic:
原始视频 → CVAT → GLINT-Bench QA

Sports / Traffic / Crowd:
原数据集 Annotation → GLINT-Bench QA
```

无论原始 Annotation 来自何处，最终均统一为 `qa.jsonl` 中的时空证据 Ground Truth。

---

## 4. `qa.jsonl`

`qa.jsonl` 用于记录每个问题对应的 Question、Answer 以及时空证据 Ground Truth（GT）。每一行对应一个 QA Sample。

当前格式如下：

```json
{
  "question_id": "magic_0001_q01",
  "video_id": "magic_0001",
  "category": "magic",
  "question_type": "mcq",
  "question": "What is the suit of the third card that appears in the person's hand? A. Hearts B. Diamonds C. Clubs D. Spades",
  "answer": "D",
  "evidence_windows": [
    {
      "start": 1.667,
      "end": 2.133
    }
  ],
  "evidence_boxes": [
    {
      "frame": 50,
      "time": 1.667,
      "box": [582.85, 1144.00, 254.05, 224.70]
    }
  ]
}
```

字段说明：

- `question_id`：问题的唯一 ID，格式为 `<video_id>_qXX`。
- `video_id`：问题对应的视频 ID。
- `category`：视频类别，目前包括 `sports`、`traffic`、`crowd`、`game` 和 `magic`。
- `question_type`：问题类型，用于支持后续不同 QA 形式的扩展。当前统一使用 `mcq`。
- `question`：问题文本。对于 MCQ，候选选项直接写在 Question 中。
- `answer`：正确答案，当前使用 `A`、`B`、`C`、`D` 等选项编号记录。
- `evidence_windows`：回答该问题所需的 Temporal Evidence GT，以秒为单位记录 `start` 和 `end`。
- `evidence_boxes`：Temporal Evidence 范围内对应的 Spatial Evidence GT。

每条 `evidence_boxes` 包含：

- `frame`：原视频中的帧号；
- `time`：该帧对应的时间戳，单位为秒；
- `box`：该帧中目标的 Bounding Box，格式统一为 `[x, y, width, height]`，使用原视频绝对像素坐标。

对于一个 Temporal Evidence 区间，`evidence_boxes` 记录该区间内有效标注帧对应的空间证据，用于后续细粒度时空证据定位评测。

### Question Type

`question_type` 字段用于支持后续不同问题形式的扩展。

当前数据构建阶段统一使用：

```text
mcq
```

即 Multiple Choice Question，以便统一计算 Answer Accuracy。

后续如需要加入开放式问答，可进一步扩展：

```text
open
```

当前版本暂不使用 `open`。
---

## 5. Metadata 生成

完成视频下载和裁剪后，可以运行：

```bash
python scripts/extract_video_metadata.py
```

脚本会扫描本地 `videos/` 目录中的视频，并自动生成：

```text
data/video_metadata.jsonl
```

因此，新增或修改 benchmark 视频后，应重新运行 metadata 提取脚本，确保 `video_metadata.jsonl` 与当前视频保持一致。
