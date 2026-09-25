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

## 3. 视频坐标系

后续空间证据标注统一使用视频自身的像素坐标系。

坐标原点位于视频帧的**左上角 `(0, 0)`**。

Bounding Box 统一表示为：

```text
[x, y, width, height]
```

其中：

* `x`：Bounding Box 左上角的横坐标；
* `y`：Bounding Box 左上角的纵坐标；
* `width`：Bounding Box 的宽度；
* `height`：Bounding Box 的高度。

所有数值均使用**绝对像素坐标**。

例如：

```json
"bbox": [412, 286, 173, 241]
```

表示 Bounding Box 左上角为 `(412, 286)`，宽度为 `173 px`，高度为 `241 px`。

因此，在进行空间标注和评测时，可以通过 `video_id` 查询 `video_metadata.jsonl`，获得对应视频的 `width` 和 `height`，从而确定 Bounding Box 所使用的坐标空间。

---

## 4. `qa.jsonl`

`qa.jsonl` 用于记录每条视频对应的问题、答案以及时空证据 Ground Truth（GT）。

**当前 QA 与 GT 标注格式尚未最终确定，本节将在标注方案确定后更新。**

后续将主要包含：

* Question
* Answer GT
* Temporal Evidence GT
* Spatial Evidence GT

具体字段结构、时间标注方式以及 Spatial GT 的组织方式将在后续版本中补充。

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
