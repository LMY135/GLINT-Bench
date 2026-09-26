# Annotation 转换脚本说明

本目录包含将 CVAT 标注转换为 GLINT-Bench `qa.jsonl` 格式所需的脚本。

```text
scripts/annotation/
├── build_qa.py
├── cvat.py
└── utils.py
```

整体流程：

```text
CVAT XML
   │
   ▼
cvat.py
解析 Track、Frame 和 Bounding Box
   │
   ▼
build_qa.py
结合 video_metadata.jsonl
生成 evidence_windows 和 evidence_boxes
   │
   ▼
utils.py
读取 / 保存 JSONL
   │
   ▼
data/qa.jsonl
```

实际使用时只需要运行 `build_qa.py`。

---

## 1. `cvat.py`

`cvat.py` 负责解析从 CVAT 导出的 XML。

CVAT 中每个 Question 对应一个 Track：

```xml
<track id="0" label="q01" source="manual">
  <box
    frame="50"
    outside="0"
    xtl="582.85"
    ytl="1144.00"
    xbr="836.90"
    ybr="1368.70">
  </box>

  ...

  <box
    frame="65"
    outside="1"
    xtl="559.75"
    ytl="1107.71"
    xbr="813.81"
    ybr="1332.41">
  </box>
</track>
```

其中：

```text
q01 → Question 01
q02 → Question 02
...
```

`cvat.py` 主要负责：

1. 读取 XML 中的 Track；
2. 根据 `outside` 判断有效标注帧；
3. 提取每一帧的 Bounding Box；
4. 将 CVAT 的 Bounding Box 格式转换为 GLINT-Bench 格式。

CVAT 使用：

```text
[xtl, ytl, xbr, ybr]
```

GLINT-Bench 使用：

```text
[x, y, width, height]
```

转换方式：

```python
x = xtl
y = ytl
width = xbr - xtl
height = ybr - ytl
```

因此：

```text
cvat.py
CVAT XML → Track + Frame + Bounding Box
```

---

## 2. `utils.py`

`utils.py` 负责 JSONL 文件的读取和保存。

其中：

```python
load_jsonl(...)
```

用于读取 JSONL。

例如：

```python
load_jsonl(path)
```

返回所有记录。

也可以通过：

```python
load_jsonl(path, key="question_id")
```

按照 `question_id` 建立索引，方便判断某个 Question 是否已经存在。

读取时还会检查：

- JSON 是否合法；
- 指定的 key 是否存在；
- 是否出现重复的 `question_id`。

```python
save_jsonl(...)
```

用于将处理后的记录重新写入 JSONL 文件。

因此：

```text
utils.py
JSONL → Load / Index / Save
```

---

## 3. `build_qa.py`

`build_qa.py` 是整个转换流程的主程序。

默认使用：

```python
ANNOTATIONS_DIR = (
    ROOT
    / "data"
    / "annotations"
)

METADATA_FILE = (
    ROOT
    / "data"
    / "video_metadata.jsonl"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "qa.jsonl"
)
```

也就是：

```text
data/annotations/*.xml
          +
data/video_metadata.jsonl
          ↓
      build_qa.py
          ↓
     data/qa.jsonl
```

脚本会：

1. 查找 `data/annotations/` 中的 CVAT XML；
2. 调用 `cvat.py` 解析 XML；
3. 根据 XML 和 Track Label 生成 `question_id`；
4. 从 `video_metadata.jsonl` 获取 FPS 和视频尺寸；
5. 将 Frame 转换为 Timestamp；
6. 生成 `evidence_windows`；
7. 生成逐帧的 `evidence_boxes`；
8. 读取已有的 `qa.jsonl`；
9. 根据 Question 是否已经存在决定新增、跳过或更新；
10. 最后通过 `utils.py` 保存结果。

---

## 4. 准备 XML

从 CVAT 下载：

```text
CVAT for video 1.1
```

然后在本地创建：

```text
data/annotations/
```

将 XML 放入该目录。

文件名需要与 `video_id` 对应：

```text
data/annotations/magic_0001.xml
data/annotations/game_0001.xml
```

同时确保对应视频已经存在于：

```text
data/video_metadata.jsonl
```

例如：

```json
{"video_id":"magic_0001","width":1080,"height":1920,"fps":30.0,"duration":15.0,"num_frames":450}
```

---

## 5. 第一次生成 QA

在项目根目录运行：

```bash
python scripts/annotation/build_qa.py
```

对于 XML 中发现的**新 Question**，脚本会自动生成：

```text
question_id
video_id
category
evidence_windows
evidence_boxes
```

但是脚本**不会自动生成 Question 和 Answer**。

因此新记录中的：

```json
"question_type": "",
"question": "",
"answer": ""
```

默认保持为空。

例如：

```json
{
  "question_id": "magic_0001_q01",
  "video_id": "magic_0001",
  "category": "magic",
  "question_type": "",
  "question": "",
  "answer": "",
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

生成后，需要人工补充：

```text
question_type
question
answer
```

例如：

```json
"question_type": "mcq",
"question": "What is the suit of the third card that appears in the person's hand? A. Hearts B. Diamonds C. Clubs D. Spades",
"answer": "D"
```

也就是说：

```text
CVAT / Script 自动生成：
question_id
video_id
category
evidence_windows
evidence_boxes

人工填写：
question_type
question
answer
```

---

## 6. 已有 Question 的跳过逻辑

默认运行：

```bash
python scripts/annotation/build_qa.py
```

时，脚本会先读取已有的：

```text
data/qa.jsonl
```

并按照：

```text
question_id
```

建立索引。

如果某个 `question_id` 已经存在，例如：

```text
magic_0001_q01
```

那么该 Question 会被**直接跳过**。

也就是说，普通运行不会重新生成已有 Question，也不会覆盖已经人工填写的：

```text
question_type
question
answer
```

同时也不会修改已有的：

```text
evidence_windows
evidence_boxes
```

因此，如果只是新增了一些 XML / Track，可以直接再次运行：

```bash
python scripts/annotation/build_qa.py
```

脚本只会添加新的 Question，已有 Question 保持不变。

如果本次运行没有新增任何 Question，`qa.jsonl` 也不会被重新写入。

---

## 7. 更新已有 Question

如果修改了已有 Question 的 CVAT 标注，例如：

- 调整了 Bounding Box；
- 修改了 Evidence 的开始帧；
- 修改了 Evidence 的结束帧；

可以使用 `--overwrite` 重新生成已有 Question 的标注信息。

`--overwrite` 支持两种更新方式：

1. **全量更新已有 Question**
2. **只更新指定 Question**

无论使用哪种方式，脚本都会保留人工填写的：

```text
question_type
question
answer
```

重新生成的主要是：

```text
video_id
category
evidence_windows
evidence_boxes
```

因此，修改 CVAT 标注后不需要重新填写 Question 和 Answer。

---

### 7.1 全量更新

如果需要根据当前 `data/annotations/` 中的 XML 重新生成所有已有 Question，运行：

```bash
python scripts/annotation/build_qa.py --overwrite
```

此时：

```text
所有已有 Question
        ↓
重新读取对应 CVAT Track
        ↓
重新生成自动字段
        ↓
保留 question_type / question / answer
        ↓
更新 qa.jsonl
```

例如：

```text
magic_0001_q01 → UPDATE
magic_0001_q02 → UPDATE
magic_0002_q01 → UPDATE
...
```

---

### 7.2 更新指定 Question

如果只修改了某一个 Question，不需要重新更新所有已有 Question。

例如只更新：

```text
magic_0001_q02
```

运行：

```bash
python scripts/annotation/build_qa.py --overwrite magic_0001_q02
```

此时：

```text
magic_0001_q01 → SKIP
magic_0001_q02 → UPDATE
其他已有 Question → SKIP
```

也可以一次指定多个 Question：

```bash
python scripts/annotation/build_qa.py --overwrite magic_0001_q01 magic_0001_q02
```

此时只会更新指定的：

```text
magic_0001_q01
magic_0001_q02
```

其他已有 Question 保持不变。

如果指定的 `question_id` 没有在当前 CVAT XML 中找到，脚本会输出 Warning，例如：

```text
[WARNING] Selected question not found in CVAT XML: magic_0001_q99
```

---

### 7.3 Overwrite 时保留的字段

假设原来的数据为：

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
  "evidence_boxes": []
}
```

修改 CVAT XML 后运行：

```bash
python scripts/annotation/build_qa.py --overwrite magic_0001_q01
```

更新逻辑为：

```text
question_id       → 保持对应 Question ID

video_id          → 根据当前标注重新生成
category          → 根据当前标注重新生成

question_type     → 保留原值
question          → 保留原值
answer            → 保留原值

evidence_windows  → 根据最新 XML 重新生成
evidence_boxes    → 根据最新 XML 重新生成
```

因此，`--overwrite` 不是简单覆盖人工 QA 内容，而是重新构建自动生成的数据，同时保护人工填写字段。

---

## 8. 三种运行模式总结

### 模式 1：普通运行

```bash
python scripts/annotation/build_qa.py
```

逻辑：

```text
新 Question
    ↓
ADD

已有 Question
    ↓
SKIP
```

适用于：

```text
新增视频 / 新增 Track / 新增 Question
```

---

### 模式 2：全量 Overwrite

```bash
python scripts/annotation/build_qa.py --overwrite
```

逻辑：

```text
新 Question
    ↓
ADD

所有已有 Question
    ↓
UPDATE
```

已有 Question 会重新生成自动字段，但：

```text
question_type
question
answer
```

保持不变。

适用于：

```text
需要重新处理当前所有 CVAT 标注
```

---

### 模式 3：指定 Question Overwrite

更新一个 Question：

```bash
python scripts/annotation/build_qa.py --overwrite magic_0001_q02
```

更新多个 Question：

```bash
python scripts/annotation/build_qa.py --overwrite magic_0001_q01 magic_0001_q02
```

逻辑：

```text
指定的已有 Question
        ↓
      UPDATE

未指定的已有 Question
        ↓
       SKIP

新 Question
        ↓
       ADD
```

适用于：

```text
只修改了部分 Question 的 CVAT 标注
```

---

## 9. `--overwrite` 参数逻辑

脚本内部通过 `args.overwrite` 的不同值区分三种模式：

```text
args.overwrite = None
```

表示没有使用：

```bash
--overwrite
```

此时已有 Question 全部跳过。

---

```text
args.overwrite = []
```

表示使用了：

```bash
--overwrite
```

但没有指定 Question ID。

此时更新所有已有 Question。

---

```text
args.overwrite = [
    "magic_0001_q01",
    "magic_0001_q02"
]
```

表示使用了：

```bash
--overwrite magic_0001_q01 magic_0001_q02
```

此时只更新指定的已有 Question。

因此可以概括为：

```text
None
→ 不更新已有 Question

[]
→ 更新全部已有 Question

[question_id, ...]
→ 只更新指定 Question
```

---

## 10. 推荐工作流程

### 新增标注

```text
从 CVAT 下载 XML
        ↓
放入 data/annotations/
        ↓
python scripts/annotation/build_qa.py
        ↓
新 Question → ADD
已有 Question → SKIP
        ↓
人工填写 question_type / question / answer
```

### 修改单个 Question 的标注

例如修改：

```text
magic_0001_q02
```

运行：

```bash
python scripts/annotation/build_qa.py --overwrite magic_0001_q02
```

流程：

```text
修改 / 重新下载 CVAT XML
        ↓
替换 data/annotations/ 中对应 XML
        ↓
指定 magic_0001_q02
        ↓
只更新该 Question
        ↓
保留原 question_type / question / answer
```

### 修改多个 Question

```bash
python scripts/annotation/build_qa.py --overwrite magic_0001_q01 magic_0001_q02
```

只更新指定的 Question。

### 需要全部重新生成 Evidence

```bash
python scripts/annotation/build_qa.py --overwrite
```

更新当前 XML 中所有已有 Question。

---

## 11. 命令速查

```bash
# 新增数据，已有 Question 跳过
python scripts/annotation/build_qa.py

# 更新所有已有 Question
python scripts/annotation/build_qa.py --overwrite

# 更新一个指定 Question
python scripts/annotation/build_qa.py --overwrite magic_0001_q02

# 更新多个指定 Question
python scripts/annotation/build_qa.py --overwrite magic_0001_q01 magic_0001_q02
```

其中，任何 Overwrite 模式都不会覆盖人工填写的：

```text
question_type
question
answer
```