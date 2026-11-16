# Evaluation 使用说明

本实现对接当前 glint.models 的 get_model()、generate()、Text、Video、Image 和 ModelResponse。
当前数据使用 MCQ、单个时间区间及原始帧像素 xywh 框。
默认 task 为 progressive_qa，执行三组独立的答案预测。task 为 grounding 时执行原有定位流程。

## Game 类别全量重跑（云端 API，Grounding）

以下示例使用已在 `configs/models.yaml` 声明的 `gemini-3.1-pro`。在 Windows PowerShell 的项目根目录执行；直接调用现有 Conda 环境的 Python，不需要激活 Conda，也不需要 CUDA。

```powershell
cd D:\GLINT-Bench
& "C:\Users\86187\.conda\envs\glint-api\python.exe" -m glint.evaluation --task grounding --model gemini-3.1-pro --categories game --output-dir results/gemini-3.1-pro/grounding_game_v2
```

- `--categories game`：筛选全部 Game 问题；**不要设置 `--limit` 或 `--video-limit`**。也请确认 `configs/evaluation.yaml` 中的这两项为 `null`，`video_ids` / `question_ids` 为 `[]`，否则 YAML 中的筛选仍会生效。
- `--task grounding`：先对原生视频预测答案及时间区间，Temporal Gate 通过后才采样帧并预测空间框。
- `--output-dir .../grounding_game_v2`：使用全新目录，不复用旧的预测和 Checkpoint；保留旧结果作对照。中断后以**完全相同的命令**恢复。
- 云端服务需要已配置 API Key 与有效网络连接；Grounding 还需要本地可执行的 `ffmpeg`、`ffprobe` 及视频文件。
- 目前 Stage Token 配额（QA 256、Video 512、Spatial 1024）**仅对本地 HuggingFace Adapter 生效**，并未应用到 Gemini / OpenAI 兼容接口；云端用其原有生成参数。
- Progressive QA 与 Grounding 是独立任务；上面的命令**不会**运行三条件 Progressive QA。

完成后查看：

```text
results/gemini-3.1-pro/grounding_game_v2/
    game/grounding/
        predictions.jsonl
        scores.jsonl
        summary.json
    .run_state/
        run_config.json
        checkpoints/
        score_config.json
```

`summary.json` 的 `overall` 和 `by_category.game` 包含 `accqa`、`tnwd`、`tg_siou`、`cs_iou`、`gate_passed`、`status_counts`，并记录每个阶段的失败状态。

### 独立离线计算 CS-IoU

直接读取已有 Grounding `scores.jsonl`，无需调用模型、重新采样或重新下载视频：

```powershell
& "C:\Users\86187\.conda\envs\glint-api\python.exe" -m glint.evaluation.cs_iou --input results/gemini-3.1-pro/grounding_game_v2 --output results/gemini-3.1-pro/cs_iou_game_v2.json
```

也可以使用 `--input results/.../game/grounding/scores.jsonl` 指向单个类别文件。CS-IoU 是通过 Temporal Gate 的样本的空间 IoU 均值；通过 Gate 但空间预测失败计为 0，没有任何样本通过 Gate 则返回 `null`。

## 一、配置说明

### 1. 安装依赖

**只跑云端 API 就跳过 PyTorch、Transformers、Accelerate 等本地推理依赖。** PyTorch 及其 CUDA 依赖下载体积较大，只有要在自己机器上运行模型时才需要安装。`requirements.txt` 包含云端 SDK、公共依赖和视频下载工具。`environment.yml` 还会安装本地推理依赖。

在仓库根目录操作。先激活计划用来运行评测的 Python 环境，再用同一个 Python 安装依赖和运行评测。

**只跑一家云端服务。** 安装公共依赖和对应的 SDK，下面两条命令任选其一。

```bash
# Gemini
python -m pip install "PyYAML>=6.0,<7" "python-dotenv>=1.0,<2" "google-genai>=1.75,<2"

# OpenAI 兼容服务
python -m pip install "PyYAML>=6.0,<7" "python-dotenv>=1.0,<2" "openai>=1.50,<3"
```

如果还需要下载视频，再安装 `yt-dlp`（`python -m pip install "yt-dlp>=2025.1,<2027"`）。

**同时跑 Gemini 和 OpenAI 兼容服务。** 可以一次性安装 `requirements.txt` 中的依赖。

```bash
python -m pip install -r requirements.txt
```

**跑本地 Hugging Face 模型。** 参考 `environment.yml` 安装 PyTorch、Transformers、Accelerate 和视频处理依赖，并按机器的 CUDA 环境选择 PyTorch 构建。需要从头创建完整环境时，可运行 `conda env create -f environment.yml`，随后运行 `conda activate glint-bench`。

`grounding` 需要系统能找到 `ffmpeg` 和 `ffprobe`。只跑 `progressive_qa` 时，评测流程不调用这两个命令。

### 2. 声明模型

为支持原生视频输入，GLINT-Bench 需要按 provider 的协议封装视频请求，因此这里不使用 LiteLLM 自动推断 provider。请在 `configs/models.yaml` 的 `models:` 下显式声明模型、provider 和输入能力。

```yaml
  gemini-3.1-pro:
    provider: cloud_gemini
    model_id: gemini-3.1-pro-preview
    supports_native_video: true
    supports_images: true
```

云端模型的 API Key 填在仓库根目录的 `.env` 中（如 `GEMINI_API_KEY`）。运行时用 `--model gemini-3.1-pro` 选择该条目。

### 3. 运行评测

在 `configs/evaluation.yaml` 中核对 `qa_path`、`metadata_path` 和 `videos_dir`，并确认要运行的模型已在 `configs/models.yaml` 中声明。配置好后，到第三部分选择对应任务的命令。第一次配置或更换视频目录时，可先运行可选的输入检查。

#### 数据和视频路径

读取规则与 scripts/prepare_videos.py 一致。

```text
videos/<category>/<video_id>.mp4
```

例如 magic_0001 对应 videos/magic/magic_0001.mp4。
传给 videos_dir 的路径应是包含 magic、game 等分类子目录的视频根目录。
QA 使用 question_id 作为主键，通过 video_id 关联 video_metadata.jsonl。
缺失视频、重复主键、无 metadata、无效标注会在模型加载前报错。
GT 时间相对于裁剪后的 benchmark 视频起点，帧号按零起始解释。
选项直接使用 question 内已有文本。

配置文件中的数据路径相对于 project_root 解析。示例 project_root 为配置文件的上一级目录。
CLI --videos-dir 和 --output-dir 相对于当前工作目录解析。

## 二、输出目录

GLINT-Bench 有两种评测。两种评测分别运行，并各自指定 `--output-dir`。下面以只跑 Magic 类别为例。跑多个类别时，每个类别都会有自己的结果目录。

测评结果去对应的任务文件，查看类别文件即可。如results/model_name/grounding/magic/

### Grounding（`--task grounding`）

评估模型能否定位回答所需的证据。模型先预测时间区间，通过时间门控后再预测采样帧上的空间框。

```text
results/model_name/grounding/        # --output-dir
    magic/
        grounding/
            predictions.jsonl
            scores.jsonl
            summary.json
    .run_state/                      # 这些都是一些runtime状态文件，不用管，index是所有的文件位置索引
```

- `predictions.jsonl` 保存逐题的模型回复、预测时间区间和空间框（若执行了空间阶段）。
- `scores.jsonl` 保存逐题的 AccQA、TNWD、Temporal Gate、sIoU 和 TG-sIoU；CS-IoU 从这些已有分数离线汇总即可。
- `summary.json` 汇总该类别的 AccQA、TNWD、TG-sIoU、CS-IoU、Temporal Gate 通过数量及状态统计。

### Progressive QA（三种证据条件，`--task progressive_qa`）

比较同一道题在三种证据条件下的答题准确率。每个条件独立生成答案。

```text
results/model_name/progressive_qa/   # --output-dir
    magic/
        summary.json
        baseline/
            predictions.jsonl
            scores.jsonl
            summary.json
        temporal_gt/
            predictions.jsonl
            scores.jsonl
            summary.json
        spatiotemporal_gt/
            predictions.jsonl
            scores.jsonl
            summary.json
    runtime/
        run_config.json
        checkpoints/
        result_index.json
```

- `baseline/` 提供完整视频和问题，不提供标注证据。
- `temporal_gt/` 在基线输入上提供真实时间区间。
- `spatiotemporal_gt/` 同时提供真实时间区间和空间框。

每个条件的 `predictions.jsonl` 保存逐题答案和正误，`scores.jsonl` 保存逐题 AccQA，`summary.json` 统计该条件。每个类别的 `summary.json` 汇总三种条件准确率；`runtime/result_index.json` 提供文件索引。

Grounding 使用 `.run_state/` 保存运行配置与 Checkpoint；Progressive QA 使用 `runtime/` 保存状态及文件索引。两种任务均可续跑。

### Progressive QA 的输入与逐题记录

configs/evaluation.yaml 默认设置 task: progressive_qa。
同一问题执行三次独立请求，每次重新构造消息，使用相同的完整原生视频。

| condition | 输入证据 | 模型输出 |
|---|---|---|
| baseline | 视频和问题 | 单个选项字母 |
| temporal_gt | 视频、问题和 GT 时间区间 | 单个选项字母 |
| spatiotemporal_gt | 视频、问题、GT 时间区间和逐帧 GT 框 | 单个选项字母 |

时间和空间证据作为文本提示加入。三种条件使用相同的视频输入，不做 GT 裁剪或额外图片输入。
空间 GT 使用标注中的 time、frame 和原始像素 xywh 坐标。
每次请求只使用当前条件的输入，正确答案及其他条件的预测保留在评分侧。
三次均只要求回答问题。模型输出示例为单个 D。

每个类别下的三个条件各有一份 predictions.jsonl，每题一行。

```json
{
  "question_id": "magic_0001_q01",
  "video_id": "magic_0001",
  "category": "magic",
  "condition": "baseline",
  "status": "ok",
  "raw_response": "D",
  "predicted_answer": "D",
  "correct": true,
  "history": []
}
```

当前 Progressive QA 代码保存 `raw_response`、`predicted_answer`、`correct`、`history` 和状态；思考摘要不作为稳定输出字段。解析失败时在 `history` 中记录错误。

summary.json 分条件、分分类汇总 AccQA，并保存 temporal_minus_baseline 和
spatiotemporal_minus_temporal 两个准确率差值。差值允许为负。
每种条件各自以所选范围内的全部问题为分母，调用或解析失败的记录计 0 并报告状态。
checkpoint 按 question_id 和 condition 保存。中断后继续时只补跑缺失条件，
--retry-failed 只重跑失败条件。此任务不需要 ffmpeg 或 ffprobe。

这个对照设计参考 VideoZeroBench 的前三种 QA 条件，其论文按证据从多到少编号为 L1、L2、L3，
本实现按 baseline、temporal_gt、spatiotemporal_gt 从少到多命名。
提示文本、结果字段和 JSONL 文件组织是 GLINT-Bench 的实现约定。
参考论文 [VideoZeroBench, Section 2.3](https://arxiv.org/html/2604.01569v3)。

### Grounding 的预测与评分

第一阶段输入原生视频和问题，返回以下 JSON。

```json
{"answer":"D","evidence_windows":[{"start":1.7,"end":2.1}]}
```

答案和时间分别解析，时间无效时仍可计算有效答案的 AccQA。

TNWD 使用仓库 results/README.md 中的公式。

```text
D_T = sqrt((c_pred - c_gt)^2 + (length_pred - length_gt)^2 / 4)
TNWD = exp(-D_T / C)
```

TNWD 达到 tau 且两个时间区间具有正长度交集时，执行空间阶段。
从交集内有标注的帧均匀选择最多 K 个不同帧，按原始帧号提取原尺寸 JPEG。
这一步选帧使用 GT 时间和标注帧可用性。模型收到问题、第一阶段预测上下文、帧标识和图片。
GT 答案及 GT 框保留在评分侧。

第二阶段返回以下结构。

```json
{"evidence_boxes":[{"frame":50,"box":[580,1140,255,225]}]}
```

同帧 IoU 取均值得到 sIoU，通过时间门控后得到 TG-sIoU。
少于 K 个可用帧时按实际帧数取均值。漏掉的预测框记 0。
交集内无有效标注帧或本地解码失败记录为 sampling_error。
第一版明确要求每题只有一个 evidence_window。

## 三、运行命令

### 1. 选择评测任务

模型和数据路径配置好后，可以直接运行评测。根据需要选择任务，并将 `--model` 替换成自己的模型条目名称。

输出目录统一写成 results/<模型名称>/<任务类型>。`<模型名称>` 与 --model 一致，`<任务类型>` 与 --task 一致，可填 progressive_qa 或 grounding。

```bash
# Progressive QA：比较同一题在三种证据条件下的答题准确率
python -m glint.evaluation --config configs/evaluation.yaml --task progressive_qa --model gemini-3.1-pro --output-dir results/gemini-3.1-pro/progressive_qa

# Grounding：评估时间定位，时间门控通过后再评估空间定位
python -m glint.evaluation --config configs/evaluation.yaml --task grounding --model gemini-3.1-pro --output-dir results/gemini-3.1-pro/grounding
```

### 2. 可选：检查本机的数据路径

第一次配置或更换视频目录时，可以用下面的命令检查 Magic 类别的第一道题。程序会读取 QA 和视频元数据，检查该题的标注格式及对应视频文件是否存在。通过时，终端会显示 `Validated 1 questions, 1 local videos`，并打印题目 ID 和解析出来的视频路径。如果数据或路径有问题，命令会报出对应错误。这一步只检查本地输入，不加载模型或调用 API，日常运行可以跳过。

```bash
python -m glint.evaluation --config configs/evaluation.yaml --categories magic --limit 1 --validate-only
```

视频存放在其他位置时，可以用 `--videos-dir` 指向包含各类别子目录的视频根目录。

```bash
python -m glint.evaluation --categories magic --limit 1 --videos-dir /data/GLINT-Bench/videos --validate-only
```

### 3. 按类别运行

通过 categories 选择需要评测的类别，默认空列表表示全部。

```yaml
categories: [magic]
```

命令行参数优先于 YAML 配置。

```bash
python -m glint.evaluation --categories magic --validate-only
python -m glint.evaluation --categories magic --output-dir results/your_model/magic
python -m glint.evaluation --categories magic game --output-dir results/your_model/magic_game
```

支持 sports、traffic、crowd、game、magic。
筛选发生在视频文件和 metadata 内容校验之前，未选中类别的本地视频缺失不会阻塞运行。
无匹配问题时提前报错。相同规则同时适用于 progressive_qa 和 grounding。
逐题结果与汇总的分母仅包含所选类别的问题，筛选条件写入 `.run_state/run_config.json`。
使用相同条件和输出目录再次运行会从 checkpoint 继续。
切换类别组合时使用独立 output_dir，已有记录不会被另一个实验范围覆盖。
例如先运行 magic，下次可运行 game 并使用 results/your_model/game。
选择类别时不需要改写 qa.jsonl。

### 4. 按 ID 或原始记录顺序运行

支持指定 qa.jsonl 中的 video_id 或 question_id，以及只运行前 N 条记录。

```bash
# 只跑 qa.jsonl 原始顺序的前两条问题
python -m glint.evaluation --limit 2 --output-dir results/your_model/first2_questions

# 只跑 qa.jsonl 中首次出现的前两个不同视频及其关联问题
python -m glint.evaluation --video-limit 2 --output-dir results/your_model/first2_videos

# 按 video_id 指定视频，运行它们关联的全部问题
python -m glint.evaluation --video-ids magic_0001 magic_0002 --output-dir results/your_model/two_videos

# 精确指定问题
python -m glint.evaluation --question-ids magic_0001_q01 magic_0001_q02 --output-dir results/your_model/two_questions

# 魔术类别筛选后的前两条问题
python -m glint.evaluation --categories magic --limit 2 --output-dir results/your_model/magic_first2
```

相同选择条件也可以写入 evaluation.yaml。

```yaml
categories: []
video_ids: []
question_ids: []
limit: 2
video_limit: null
```

空 ID 列表表示所有 ID，null 表示不限制条数。两个条数参数均要求正整数。
类别、video_ids 和 question_ids 先取交集，再取前 video_limit 个不同视频，最后取前 limit 条 QA。
原始顺序是 qa.jsonl 中非空记录的出现顺序，不按 ID 排序。
即使同一视频的问题分散在文件的不同位置，video_limit 也会保留该视频全部符合条件的问题。
显式 ID 在 QA 文件中不存在时提前报错，避免只执行部分拼写正确的 ID。
所有筛选均在本地视频和 metadata 内容校验之前完成。
`.run_state/run_config.json` 保存选择条件，同一范围可以恢复执行；改变范围时使用新的 output_dir。
两道问题的渐进式 QA 正常产生 2×3=6 条回答记录，汇总分母每种条件都是 2。
这里 video_id 指 benchmark clip ID，例如 magic_0001。

### 5. 续跑、重试和重新评分

**不同任务的解析失败策略不同：** Progressive QA 的单选答案至少允许一次额外生成（如果配置的 `parse_retries` 更高，则使用其值）；Grounding 的 Video 阶段按 `parse_retries` 控制额外生成，而 **Spatial 阶段遇到解析错误不会自动重生成**。Spatial 会保存原始回复及错误信息；`--retry-failed` 会尝试用改进后的解析器重新解析已有回复。仍然无法解析的空间输出维持失败状态，不会重新调用 Spatial API。要使用更新 Prompt 对这些样本重新推理，请新建 `--output-dir`。

max_retries 为额外调用重试次数，默认 2，最多进行 3 次模型调用。
超时、连接错误、408、429、500、502、503、504 按指数退避和随机抖动重试。
鉴权、参数及确定性解码错误直接记录失败。
parse_retries 默认为 0，仅用于允许解析重试的阶段。可恢复的 API/网络错误仍由 Adapter 的传输重试机制处理；它与解析错误的重试策略不同。
空框列表可被正常解析，各采样帧均记 0。

按阶段原子保存 checkpoint，中断后再次运行相同命令即可继续。
对于 QA 及适用的 Grounding 阶段，可使用 --retry-failed 处理失败；Grounding Spatial 解析失败只重新解析历史文本，不自动重新调用模型。
时间门控未通过属于正常完成。

下面以已有的 Grounding 运行目录为例。续跑、重试或重新评分时，沿用原运行的任务、模型、筛选条件和输出目录。如果原命令用了 `--categories` 或 `--limit`，这里也要带上同样的值。

```bash
# 重试此前失败的样本
python -m glint.evaluation --config configs/evaluation.yaml --task grounding --model gemini-3.1-pro --output-dir results/gemini-3.1-pro/grounding --retry-failed

# 用已有预测重新评分
python -m glint.evaluation --config configs/evaluation.yaml --task grounding --model gemini-3.1-pro --output-dir results/gemini-3.1-pro/grounding --score-only
```

所选范围内的全部问题计入汇总分母。调用或解析失败时，受影响指标记 0，已成功解析的其他字段继续评分。
每个类别的 `summary.json` 汇总该类别的指标和状态数量。

默认再次运行会检查数据、模型配置、Prompt 版本、视频路径/大小/修改时间及推理参数是否一致。
改变这些配置需要新的 output_dir。视频身份检查使用文件 stat，数据和模型配置使用 SHA-256。
grounding 重新评分时允许仅改变 C 或 tau，新运行的实际评分参数写入 `.run_state/score_config.json`。
若新门控需要原先被跳过的空间推理，重新评分会报错，需要用新的输出目录运行完整评测。
K 改变会影响空间输入，需要新运行。

每个输出目录同时只允许一个写入进程。进程被强制终止后若遗留 .run.lock，确认没有任务运行后删除该空目录。
模型实际调用的联网验证需要用户配置 API Key 或本地模型权重。

### 6. 开发者回归测试

仓库中已有 `tests/` 时，可以运行本地回归测试。

```bash
python -m pip install "pytest>=8,<10"
python -m pytest -q
```

测试覆盖本地目录、外部视频根目录、缺失文件、原始视频误匹配、数据校验、TNWD 公式、时间门控、
独立字段解析、有限重试、漏框记分，以及使用合成视频实际解码的断点恢复流程。
渐进式 QA 测试覆盖提示隔离、三份预测记录、分条件评分、部分失败和按条件断点恢复。
