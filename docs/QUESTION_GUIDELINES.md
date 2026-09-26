# Question Construction Principle

GLINT-Bench 的问题构建主要遵循两个原则：

1. **Temporal Localization**：问题首先需要模型定位视频中的特定时间或事件，而不是仅依赖整段视频的全局信息。
2. **Transient Evidence**：答案应依赖短时间内出现或发生变化的瞬时视觉证据。模型需要在定位到相关时间后，进一步识别该时刻的目标及其状态。

整体过程为：

```text
Temporal Localization
        ↓
Transient Evidence
        ↓
Answer
```

为增加问题的多样性，目前主要考虑以下几类 Question Pattern。

## 1. 顺序 + 属性（Ordinal Event + Attribute）

根据事件发生的顺序定位目标（时序），再判断其视觉属性（空间）。

例如，在魔术视频中：

```text
What is the suit of the third card shown in the person's hand?

A. Hearts
B. Diamonds
C. Clubs
D. Spades
```

在体育视频中，可以选择一次持续时间较短的持球事件，例如：

```text
What is the jersey color of the third player to gain possession of the ball?

A. Red
B. Blue
C. White
D. Black
```

## 2. 状态变化（State Transition）

先定位一个特定状态（时序），再判断紧接着发生的状态变化（空间）。

例如，在魔术视频中：

```text
After the first diamond card appears in the person's hand,
what is the suit of the next card?

A. Hearts
B. Diamonds
C. Clubs
D. Spades
```

（待补充）

以上类型主要用于提供构题参考，并不要求每个视频全部覆盖。同一视频中的问题应尽量采用不同的时间关系和推理方式，避免仅通过修改序数词生成大量相似问题。

当前问题统一采用 `mcq` 格式，以便进行统一的 Answer Accuracy 评测。

## 注意事项

1. **问题描述应清晰、自然。**  
   Question 应使用自然且明确的语言描述目标事件、时间关系和目标对象，避免由于指代不清、事件描述模糊或存在多种解释而引入额外干扰。问题的难度应来自**时序定位和瞬时证据理解**，而不是语言歧义。

2. **MCQ 统一设置 4 个选项。**  
   每个问题应提供 **4 个候选答案（A / B / C / D）**，且仅有一个正确答案。避免使用二选一问题，因为随机猜测即可达到 50% Accuracy。

3. **选项应具有合理干扰性。**  
   Distractors 应尽量与问题和视频内容相关，避免加入明显不可能的选项。例如询问扑克牌花色时，应使用 Hearts、Diamonds、Clubs、Spades，而不是加入与扑克牌无关的类别。

4. **避免通过问题直接泄露答案或空间位置。**  
   可以使用必要的目标描述来保证问题语义清楚，例如 `the card in the person's hand`；但应避免提供不必要的空间提示，例如目标位于画面左侧、右侧或具体区域。

5. **避免重复构造高度相似的问题。**  
   同一视频中的问题应尽量采用不同的时间关系和推理方式，而不是仅修改序数词，例如连续询问 second / third / fourth card。
