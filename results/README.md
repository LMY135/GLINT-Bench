# GLINT-Bench 评测指标说明

GLINT-Bench 不仅评估模型是否能够正确回答问题，还进一步评估模型是否能够定位真正支持答案的**细粒度时空证据**。

整体评测分为三个层次：

```text
答案正确性
    ↓
时间证据定位
    ↓
时空证据定位
```

分别回答三个问题：

1. **答案是否正确？**
2. **是否定位到了正确的瞬时时间证据？**
3. **是否进一步定位到了该时刻真正支持答案的视觉目标？**

---

## 1. 答案准确率（AccQA）

`AccQA` 用于衡量模型是否正确回答问题。

当前 GLINT-Bench 统一采用选择题，因此直接比较模型预测选项与 Ground Truth Answer：

\[
Acc_{QA}
=
\frac{N_{\text{correct}}}{N_{\text{total}}}
\]

该指标只衡量最终答案是否正确，并不考虑模型是否真正找到了支持答案的视觉证据。

因此，在 Answer Accuracy 之外，我们进一步评估模型的 Temporal Evidence 和 Spatial Evidence。

---

## 2. 瞬时时间证据定位（TNWD）

GLINT-Bench 重点关注持续时间非常短的**瞬时证据（Instantaneous Evidence）**。

对于一个时间区间：

\[
T=[t_s,t_e]
\]

使用两个量表示：

### 时间中心

\[
c=\frac{t_s+t_e}{2}
\]

表示该证据在视频中的**时间位置**。

### 持续时间

\[
l=t_e-t_s
\]

表示该证据持续了**多长时间**。

GLINT-Bench 使用 **Temporal Normalized Wasserstein Distance（TNWD）** 评估瞬时时间证据定位。

---

### 2.1 为什么不直接使用 Temporal IoU？

传统 Temporal IoU（tIoU）根据预测时间区间与 Ground Truth 时间区间的重叠程度计算：

\[
tIoU(P,G)
=
\frac{|P\cap G|}{|P\cup G|}
\]

其中：

- \(P\) 为模型预测的时间区间；
- \(G\) 为 Ground Truth 时间区间。

对于普通的长时间事件，tIoU 可以较好地衡量两个时间区间的重叠程度。

但对于 GLINT-Bench 中的**瞬时证据**，Ground Truth 通常非常短，而 Video MLLM 给出的时间定位往往更加粗粒度，预测区间可能明显长于真实证据。

例如：

```text
GT:                 |--|
Prediction:   |----------------|
```

当 Prediction 完全覆盖 GT 时：

\[
tIoU(P,G)
=
\frac{|G|}{|P|}
\]

由于瞬时证据满足：

\[
|G|\ll|P|
\]

即使模型预测的时间范围包含了正确事件，tIoU 仍然可能非常小。

更重要的是，此时 tIoU 很容易被**预测区间的长度**主导。

例如两个长度相同的预测：

```text
Prediction A:   |----------------|
GT:                    |--|

Prediction B:             |----------------|
GT:                    |--|
```

Prediction A 的中心可能非常接近真实瞬时事件，而 Prediction B 的中心已经发生明显偏移。

但如果两个预测区间都覆盖了短暂的 GT，并且预测长度相同，它们可能得到相同或非常接近的 tIoU。

因此，在瞬时证据场景下，大量粗粒度预测可能集中在较低的 tIoU 数值范围内，使 tIoU 难以充分反映模型**究竟把事件定位在了哪里**。

对于 GLINT-Bench，我们认为时间定位中有两个信息都非常重要：

1. **Temporal Center**：模型是否找到了正确的瞬时时刻；
2. **Temporal Duration**：模型是否能够将证据范围缩小到合适的持续时间，而不是返回一个很大的时间窗口。

因此，我们希望时间定位指标能够同时考虑**中心位置**和**持续时间**。

---

### 2.2 Temporal Normalized Wasserstein Distance（TNWD）

对于模型预测区间 \(P\) 和 Ground Truth 区间 \(G\)，定义 Temporal Distance：

\[
D_T(P,G)
=
\sqrt{
(c_P-c_G)^2
+
\frac{(l_P-l_G)^2}{4}
}
\]

其中：

- \(c_P\)、\(c_G\)：预测区间和 GT 区间的时间中心；
- \(l_P\)、\(l_G\)：预测区间和 GT 区间的持续时间。

随后将距离转换为归一化相似度：

\[
TNWD(P,G)
=
\exp
\left(
-\frac{D_T(P,G)}{C}
\right)
\]

其中 \(C\) 用于控制时间尺度。

TNWD 同时考虑：

```text
中心位置偏差
     +
持续时间偏差
     ↓
    TNWD
```

其中：

\[
(c_P-c_G)^2
\]

衡量模型**是否找到了正确的时间位置**；

而：

\[
\frac{(l_P-l_G)^2}{4}
\]

衡量模型预测的时间范围**是否足够细粒度**。

因此，一个模型不能仅仅通过预测一个很大的时间窗口覆盖 GT 来获得较高分数。

同时，即使两个预测窗口长度相同，TNWD 也能够根据它们与真实瞬时事件之间的中心距离区分其定位质量。

TNWD 越接近 `1`，表示预测的时间位置和持续时间越接近 Ground Truth。

当预测区间与 Ground Truth 完全一致时：

\[
TNWD=1
\]

---

## 3. 瞬时时空证据定位（Instantaneous Spatio-Temporal Grounding）

Temporal Grounding 只能判断模型是否找到了正确的时间区域，但不能说明模型是否真正找到了该时间区域内支持答案的视觉目标。

因此，在完成时间定位后，GLINT-Bench 进一步进行 Spatial Grounding。

整体过程为：

```text
Predicted Temporal Interval
            +
Ground Truth Temporal Interval
            ↓
     Temporal Intersection
            ↓
      Sample K Frames
            ↓
   Spatial Target Localization
            ↓
     Predicted Bounding Box
            ↓
       Spatial IoU
```

### 3.1 Temporal Intersection

设 Ground Truth 时间区间为 \(T\)，模型预测时间区间为 \(\hat{T}\)。

首先计算二者的时间交集：

\[
T_{\cap}=T\cap\hat{T}
\]

Spatial Grounding 只在这个共同时间区域内进行。

这样做的原因是：空间定位只有在模型已经定位到与 Ground Truth 相关的时间区域时才具有意义。

如果：

\[
T_{\cap}\neq\emptyset
\]

则从交集区域中采样 \(K\) 个 Frame，进行第二轮 Spatial Grounding。

如果：

\[
T_{\cap}=\emptyset
\]

则说明模型预测的时间范围与真实证据完全没有重合，因此跳过第二轮 Spatial Grounding，不计算 `sIoU`。

---

### 3.2 Spatial IoU（sIoU）

对于交集区域中采样得到的 \(K\) 个 Frame，模型需要分别预测支持答案的目标 Bounding Box。

对于第 \(i\) 个 Frame：

- \(\hat{b}_i\)：模型预测的 Bounding Box；
- \(b_i\)：对应的 Ground Truth Bounding Box。

首先计算每个 Frame 上的 Bounding Box IoU：

\[
IoU(\hat{b}_i,b_i)
\]

随后对 \(K\) 个采样 Frame 的 IoU 取平均：

\[
sIoU
=
\frac{1}{K}
\sum_{i=1}^{K}
IoU(\hat{b}_i,b_i)
\]

因此，`sIoU` 衡量的是：

> 在模型和 GT 都认为相关的时间区域内，模型能否进一步找到正确的视觉目标。

`sIoU` 越高，说明模型的 Spatial Grounding 越准确。

---

### 3.3 Temporal-Gated Spatial IoU（TG-sIoU）

仅仅存在 Temporal Intersection 并不足以说明模型已经正确定位了时间证据。

例如：

```text
GT:                    |--|
Prediction:   |------------------------|
```

一个非常宽的预测区间可能包含 GT，从而产生 Temporal Intersection，但这并不能说明模型真正准确定位了瞬时事件。

因此，GLINT-Bench 使用前面的 `TNWD` 作为 Temporal Gate。

设阈值为：

\[
\tau
\]

只有同时满足：

\[
TNWD(\hat{T},T)\ge\tau
\]

以及：

\[
T\cap\hat{T}\neq\emptyset
\]

模型的 Spatial Grounding 才会获得分数。

最终定义：

\[
TG\text{-}sIoU
=
\begin{cases}
sIoU,
&
TNWD(\hat{T},T)\ge\tau
\ \text{and}\
T\cap\hat{T}\neq\emptyset
\\
0,
&
\text{otherwise}
\end{cases}
\]

因此，`TG-sIoU` 同时要求模型：

1. **时间位置正确**；
2. **时间范围足够准确**；
3. **在正确时间区域中找到正确的视觉目标**。

可以理解为：

```text
TNWD 达标？
    │
    ├── No  → TG-sIoU = 0
    │
    └── Yes
         ↓
    与 GT 有交集？
         │
         ├── No  → TG-sIoU = 0
         │
         └── Yes
              ↓
         Sample K Frames
              ↓
          Spatial IoU
              ↓
        TG-sIoU = sIoU
```

`TG-sIoU` 因此是 GLINT-Bench 的主要联合时空定位指标。

---

### 3.4 Conditional Spatial IoU（CS-IoU）

`TG-sIoU` 有意将 Temporal Grounding 和 Spatial Grounding 联合起来。

因此，如果模型的 `TG-sIoU` 很低，我们无法仅从这个指标判断：

- 是因为时间定位错误；
- 还是时间定位正确，但 Spatial Grounding 错误。

为了单独观察模型在**正确时间上下文下的空间定位能力**，GLINT-Bench 额外报告 `Conditional Spatial IoU（CS-IoU）`：

\[
CS\text{-}IoU
=
\mathbb{E}
\left[
sIoU
\mid
TNWD(\hat{T},T)\ge\tau,
\;
T\cap\hat{T}\neq\emptyset
\right]
\]

也就是说，CS-IoU **只统计 Temporal Grounding 已经通过 TNWD Gate，并且存在可评估时间交集的样本**。

因此：

- `TG-sIoU`：模型整体的**时空联合定位能力**；
- `CS-IoU`：在时间已经定位正确的情况下，模型的**纯 Spatial Grounding 能力**。

例如：

```text
Model A:
Temporal Grounding  ✗
Spatial Grounding   -
→ TG-sIoU = 0

Model B:
Temporal Grounding  ✓
Spatial Grounding   ✗
→ TG-sIoU 低
→ CS-IoU 低

Model C:
Temporal Grounding  ✓
Spatial Grounding   ✓
→ TG-sIoU 高
→ CS-IoU 高
```

通过同时报告 `TNWD`、`TG-sIoU` 和 `CS-IoU`，可以进一步区分模型的问题究竟来自 Temporal Localization 还是 Spatial Localization。

---

## 4. 指标总结

| 指标 | 评测内容 |
|---|---|
| `AccQA` | 最终答案是否正确 |
| `TNWD` | 时间证据的中心位置和持续时间是否准确 |
| Spatio-Temporal Grounding | 在相关瞬时事件中是否进一步定位到正确的视觉目标 |

GLINT-Bench 的评测标准逐渐变得更加严格：

```text
答对问题
   ↓
找对时间
   ↓
找对视觉证据
```

因此，GLINT-Bench 不仅关注模型**是否答对**，还关注模型是否真正：

> **在正确的时间，看到了正确的视觉证据。**