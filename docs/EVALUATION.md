# Evaluation 使用说明

本实现对接当前 glint.models 的 get_model()、generate()、Text、Video、Image 和 ModelResponse。
当前数据使用 MCQ、单个时间区间及原始帧像素 xywh 框。
默认 task 为 progressive_qa，执行三组独立的答案预测。task 为 grounding 时执行原有定位流程。

## 安装与模型配置

在仓库根目录运行以下命令。

```bash
pip install -r requirements.txt
```

系统需要 ffmpeg 和 ffprobe。本地模型的 torch、transformers、accelerate 等依赖见 environment.yml。
在 configs/models.yaml 中填写已验证支持原生视频和图片的模型，配置 .env 中的鉴权变量。
在 configs/evaluation.yaml 中设置 model，使其对应 models.yaml 中的模型名称。
生成参数通过 generation.<backend> 传入现有 SDK 接口。

云端 provider 的 timeout 单位为秒，Gemini 在构造 SDK 客户端时转换为毫秒。
示例模型配置设置 max_retries 为 0，由 Evaluation 控制传输重试。
自定义 provider 配置也应设置 max_retries 为 0，避免 SDK 内部重试叠加。
本地推理使用模型本身支持的生成参数，当前没有强制终止 GPU 推理的进程超时机制。

## 本地视频目录

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

```bash
python -m glint.evaluation --validate-only
python -m glint.evaluation --validate-only --videos-dir /data/GLINT-Bench/videos
python -m glint.evaluation --model your_model --videos-dir /data/GLINT-Bench/videos --output-dir results/your_model
```

## 结果目录与统计入口

每次运行在 output_dir 下按类别和实验条件保存结果。
例如 output_dir 为 results/model_name/progressive_qa 时，目录结构如下。

```text
results/model_name/progressive_qa/
    run_config.json
    result_index.json
    summary.json
    scores.jsonl
    checkpoints/
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
    game/
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
```

每个叶目录的 predictions.jsonl 保存该类别、该条件的逐题预测，scores.jsonl 保存对应逐题分数。
叶目录 summary.json 包含 category、condition、count、correct、accqa、status_counts。
类别目录下的 summary.json 汇总该类别的三个实验，运行根目录 summary.json 汇总本次运行所有类别。
整体准确率按逐题分数重新计算，使用样本数量加权。
只选择某个类别时仅生成它的分类目录。

result_index.json 列出本次运行所有叶目录结果的相对路径，作为后续统计脚本的统一入口。
例如一个 groups 条目如下。

```json
{
  "category": "magic",
  "condition": "baseline",
  "count": 2,
  "predictions": "magic/baseline/predictions.jsonl",
  "scores": "magic/baseline/scores.jsonl",
  "summary": "magic/baseline/summary.json"
}
```

统计脚本可遍历 groups 并读取相应 scores 路径，避免同时读入根目录合并文件而重复计数。
根目录继续保存三份 predictions_<condition>.jsonl，供已有读取代码使用。
独立 grounding 任务也按类别保存，叶目录为 <category>/grounding/，并生成相同结构的 result_index.json。
同一筛选范围可恢复运行，不同筛选范围仍使用各自 output_dir。

## 按类别运行

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
逐题结果与汇总的分母仅包含所选类别的问题，筛选条件写入 run_config.json。
使用相同条件和输出目录再次运行会从 checkpoint 继续。
切换类别组合时使用独立 output_dir，已有记录不会被另一个实验范围覆盖。
例如先运行 magic，下次可运行 game 并使用 results/your_model/game。
选择类别时不需要改写 qa.jsonl。

## 按 ID 或原始记录顺序运行

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
run_config.json 保存选择条件，同一范围可以恢复执行；改变范围时使用新的 output_dir。
两道问题的渐进式 QA 正常产生 2×3=6 条回答记录，汇总分母每种条件都是 2。
这里 video_id 指 benchmark clip ID，例如 magic_0001。

## 渐进式 QA 对比

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

程序分别保存以下三个文件，每个文件每题一行。

```text
predictions_baseline.jsonl
predictions_temporal_gt.jsonl
predictions_spatiotemporal_gt.jsonl
```

结果记录示例为程序构建的预测日志。

```json
{
  "question_id": "magic_0001_q01",
  "video_id": "magic_0001",
  "category": "magic",
  "model_name": "your_model",
  "condition": "baseline",
  "status": "ok",
  "raw_response": "D",
  "predicted_answer": "D",
  "correct": true,
  "history": [{"text": "D", "model": "model-id", "usage": null, "attempts": 1}]
}
```

scores.jsonl 以 question_id 和 condition 联合标识每次评分。
summary.json 分条件、分分类汇总 AccQA，并保存 temporal_minus_baseline 和
spatiotemporal_minus_temporal 两个准确率差值。差值允许为负。
每种条件各自以所选范围内的全部问题为分母，调用或解析失败的记录计 0 并报告状态。
checkpoint 按 question_id 和 condition 保存。中断后继续时只补跑缺失条件，
--retry-failed 只重跑失败条件。此任务不需要 ffmpeg 或 ffprobe。

这个对照设计参考 VideoZeroBench 的前三种 QA 条件，其论文按证据从多到少编号为 L1、L2、L3，
本实现按 baseline、temporal_gt、spatiotemporal_gt 从少到多命名。
提示文本、结果字段和 JSONL 文件组织是 GLINT-Bench 的实现约定。
参考论文 [VideoZeroBench, Section 2.3](https://arxiv.org/html/2604.01569v3)。

```bash
python -m glint.evaluation --task progressive_qa --model your_model --output-dir results/your_model/progressive_qa
```

## 独立的证据定位评测

在配置中设置 task: grounding，或者通过命令行选择此任务，并使用独立输出目录。

```bash
python -m glint.evaluation --task grounding --model your_model --output-dir results/your_model/grounding
```

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

## 重试与结果

max_retries 为额外调用重试次数，默认 2，最多进行 3 次模型调用。
超时、连接错误、408、429、500、502、503、504 按指数退避和随机抖动重试。
鉴权、参数及确定性解码错误直接记录失败。
parse_retries 控制解析错误后的额外生成次数，默认 0。
每次解析重试都使用相同输入重新生成，并有独立的传输重试预算。
空框列表可被正常解析，各采样帧均记 0。

grounding 结果目录包含 run_config.json、score_config.json、predictions.jsonl、scores.jsonl、summary.json，
以及 checkpoints/ 和 frames/。原始回复、usage、各阶段调用次数与错误保存在预测记录中。
按阶段原子保存 checkpoint，中断后再次运行相同命令即可继续。
调用失败或解析失败的样本通过 --retry-failed 重新运行。
时间门控未通过属于正常完成。

```bash
python -m glint.evaluation --config configs/evaluation.yaml --retry-failed
python -m glint.evaluation --config configs/evaluation.yaml --score-only
```

所选范围内的全部问题计入汇总分母。调用或解析失败时，受影响指标记 0，已成功解析的其他字段继续评分。
summary.json 同时给出整体、类别及最终状态数量。

默认再次运行会检查数据、模型配置、Prompt 版本、视频路径/大小/修改时间及推理参数是否一致。
改变这些配置需要新的 output_dir。视频身份检查使用文件 stat，数据和模型配置使用 SHA-256。
grounding 重新评分时允许仅改变 C 或 tau，实际评分参数写入 score_config.json。
若新门控需要原先被跳过的空间推理，重新评分会报错，需要用新的输出目录运行完整评测。
K 改变会影响空间输入，需要新运行。

每个输出目录同时只允许一个写入进程。进程被强制终止后若遗留 .run.lock，确认没有任务运行后删除该空目录。
模型实际调用的联网验证需要用户配置 API Key 或本地模型权重。

## 验证

```bash
pip install pytest
python -m pytest -q
```

测试覆盖本地目录、外部视频根目录、缺失文件、原始视频误匹配、数据校验、TNWD 公式、时间门控、
独立字段解析、有限重试、漏框记分，以及使用合成视频实际解码的断点恢复流程。
渐进式 QA 测试覆盖提示隔离、三份预测记录、分条件评分、部分失败和按条件断点恢复。
