# AlphaGen 旧链奖励设计技术报告（基于 2026-04-01 当前实验）

## 1. 目标与结论摘要

本文档讨论旧链 PPO 公式生成框架中奖励设计的合理性，并基于当前已经完成的 36 个 run 的历史泛化评估结果，对不同奖励设计进行比较。

当前阶段可以先给出三个明确结论：

1. `RE` 必须是主奖励，原因是它直接对应“新因子加入池后是否真的改善组合表现”，这是任务目标本身，而不是代理目标。
2. `RI_struct` 是目前最有证据支持的辅助项。无论在 V1 还是 V2，带结构探索项的配置都稳定进入前列，说明“结构层面的探索引导”是有价值的。
3. `re_v2+all` 目前还不能作为默认主线替代 `re`。它在当前结果中整体波动更低，但在主要效果指标 `post_rankic_mean` 上并没有超过基线 `re`。

因此，当前最合理的技术判断不是“V2 全面优于 V1”，而是：

- 奖励主项继续以 `RE` 为核心；
- 结构探索奖励值得保留并继续强化；
- 功能增量奖励与复杂度正则的实现还需要额外实验验证；
- `re_v2+all` 的完整组合式奖励还未达到接管主线的证据门槛。

## 2. 问题定义与奖励设计逻辑

### 2.1 任务本质

旧链任务不是“生成一个看起来新奇的表达式”，也不是“生成一个单独 IC 高的因子”，而是：

> 在已有因子池条件下，搜索一个能带来真实边际增益的新因子。

因此奖励设计必须尽量围绕“条件边际贡献”来组织。

### 2.2 为什么 `RE` 必须是主项

`RE` 可以理解为新因子加入当前池之后，组合效果的真实变化。无论具体实现是 `ensemble` 还是其他组合口径，它都比任何代理项更接近最终目标。

这是因为：

1. 高单因子 IC 不等于高组合增益。
2. 与已有池低相关也不等于有用。
3. 结构上新颖也不等于有效。

只有 `RE` 真正回答了问题：

> 这个新表达式加入现有池以后，组合是否更好了？

从任务对齐角度，`RE` 是一阶目标；其他 RI 项都只能是辅助项。

### 2.3 为什么需要 RI，而不是只用 RE

只用 `RE` 的问题是搜索过程容易过早收敛到局部结构模式。尤其在表达式空间很大、奖励噪声较强时，策略会偏向重复利用已经验证过的表达式模板。

因此需要 RI 类项做两件事：

1. 给探索提供方向。
2. 对复杂度与风险进行约束。

但 RI 不能喧宾夺主，否则就会从“找有效因子”变成“找看起来不同的因子”。

这也是 V2 奖励设计的基本原则：

> RI 是 bonus / regularizer，不是替代 RE 的目标函数。

## 3. 当前代码口径

当前旧链奖励在代码中是如下结构：

```text
reward = RE + lambda_t * (
  alpha * RI_func +
  beta  * RI_struct +
  gamma * RI_reg
)
```

其中：

- V1：`lambda_t = lambda_ri`
- V2：`lambda_t = lambda_ri / (1 + ri_schedule_decay * eval_cnt)`

对应实现位置：

- 奖励入口：[alpha_pool.py](C:/Users/qdz/Desktop/开题报告/alphagen/alphagen/models/alpha_pool.py)
- 训练入口参数：[train_maskable_ppo.py](C:/Users/qdz/Desktop/开题报告/alphagen/train_maskable_ppo.py)
- 历史泛化评估：[posthoc_eval_runs.py](C:/Users/qdz/Desktop/开题报告/alphagen/scripts/posthoc_eval_runs.py)

### 3.1 V1 的逻辑

V1 中：

- `RI_func` 本质是 mutual IC 的负项，即“和已有池越相似越惩罚”。
- `RI_struct` 基于 token bigram 的 Jaccard 差异。
- `RI_reg` 主要是表达式长度惩罚。

这种设计的优点是简单、可解释、计算成本低。缺点也很明显：

1. “低相关”不是“条件增量”。
2. “远离已有结构”不一定带来价值。
3. 长度惩罚过于粗糙，不能充分表达表达式风险。

### 3.2 V2 的逻辑

V2 的修改方向是正确的，主要体现在三点：

#### `RI_func_v2`

不再直接惩罚与池内因子的相关性，而是：

1. 用池内若干重要因子拟合候选因子；
2. 取残差；
3. 再计算残差对目标收益的相关性。

这更接近“条件剩余信息”的定义。理论上它比 mutual IC 更符合“新增信息”这一目标。

#### `RI_struct_v2`

不再把“新”本身当奖励，而是：

1. 先用 token bigram 做轻量结构簇划分；
2. 对每个簇维护历史 `mean_re` 与命中次数；
3. 对“历史有价值且当前覆盖不足”的簇给 bonus。

这比单纯“远离已有簇”更合理，因为它把结构探索和历史收益联系了起来。

#### `RI_reg_v2`

扩展为：

- 长度惩罚
- 深度惩罚
- 高风险算子惩罚

这比只惩罚 token 长度更接近实际表达式风险。

## 4. 当前实验设置

### 4.1 训练口径

本轮报告基于旧链结果，训练数据使用：

- 市场：`tcsi300`
- 数据路径：`tcsi300_kaggle_subset_20260326_162656`
- 训练窗口：`2014-2018`

### 4.2 泛化评估口径

历史泛化评估按年进行，划分为：

- `pre`：`2005-2013`
- `post`：`2019-2022`

每个 checkpoint step 都会计算：

- `year_ic`
- `year_rankic`
- 年度聚合后的均值 / 标准差 / 方差

本轮可视化和汇总主要使用：

1. `post_rankic_mean`
2. `all_rankic_std`
3. `generalization_gap = post_rankic_mean - pre_rankic_mean`

### 4.3 当前样本规模

当前 `data/runs` 下共完成 36 个 run 的评估。

其中存在 4 组重复的 `re` baseline（同一个 `seed × backbone × reward_mode` 被重复下载，但指标一致），因此：

- 运行记录是 36 条；
- 唯一配置数是 32 条；
- 对 `reward_mode` 做总体均值时，`re` 被轻微过采样。

所以后续严谨比较时，应优先看：

1. `reward_mode × backbone` 的分组结果；
2. 或者显式按 `(seed, backbone, reward_mode)` 去重后的结果。

## 5. 指标合理性分析

### 5.1 哪些指标是合理的

#### `post_rankic_mean`

这是当前最重要的效果指标。因为它只看训练后年份，能够回答：

> 该奖励设计是否真的提升了未来时期的横截面预测能力？

这应该作为主排序指标。

#### `all_rankic_std`

这反映跨年份的整体波动程度。它不是收益指标，但对稳定性非常重要。

当前阶段把它作为次指标是合理的。

#### `generalization_gap`

它可以衡量训练前后两个时代之间的落差，有助于判断是否存在明显的时变失效。

但它只能作为辅助解释项，不能单独作为主排序指标。因为 gap 小可能来自：

1. pre 和 post 都高；
2. pre 和 post 都低；
3. pre 被压低导致差值看起来更平。

### 5.2 哪些指标当前还不够合理

#### `best_step_by_post_rankic`

它当前用于总结时是有信息泄漏的。因为它直接在 `post` 年份上选最佳 step，本质上是“用测试集选模型”。

这可以用于：

- 事后分析；
- 观察上界；

但不能作为严格的部署结论或论文主表指标。

更严谨的做法应该是：

1. 用训练内验证信号选 step；
2. 或固定 final step；
3. 再报告 `post` 表现。

#### 训练日志中的 `best_rankic / best_ic`

这些是训练期池内指标，更多反映搜索过程内部状态，不能直接代表跨时间泛化。

它们适合做训练监控，不适合作为方法优劣的主结论。

## 6. 当前实验结果

### 6.1 Backbone 结论

按当前 36 个 run 汇总：

- `transformer` 的平均 `post_rankic_mean` 为 `0.09635`
- `lstm` 的平均 `post_rankic_mean` 为 `0.09232`

这说明在当前表达式生成任务里，`transformer` backbone 整体强于 `lstm`。

这点结论比较稳定，后续主实验建议优先以 `transformer` 为主。

### 6.2 `reward_mode × backbone` 排名

当前前五名为：

1. `re_v2+struct × transformer`：`0.09989`
2. `re × transformer`：`0.09884`
3. `re_v2+all × transformer`：`0.09744`
4. `re_v2+reg × lstm`：`0.09718`
5. `re+reg × transformer`：`0.09650`

这组结果说明：

1. 结构项确实可能有效，尤其在 `transformer` 上；
2. 基线 `re` 仍然非常强，不是容易被替代的弱基线；
3. `re_v2+all` 并没有显著拉开与 `re` 的差距。

### 6.3 V1 vs V2（按 backbone 聚合）

按 backbone 汇总 V1 / V2：

#### `lstm`

- V1 `post_rankic_mean`：`0.09241`
- V2 `post_rankic_mean`：`0.09250`
- 差值：`+0.00009`

说明在 `lstm` 上，V2 整体效果与 V1 基本持平。

#### `transformer`

- V1 `post_rankic_mean`：`0.09679`
- V2 `post_rankic_mean`：`0.09528`
- 差值：`-0.00151`

说明在 `transformer` 上，V2 当前整体效果略弱于 V1。

但与此同时：

- V2 的 `all_rankic_std` 更低

这说明 V2 在 `transformer` 上有“更稳但略弱”的倾向。

### 6.4 `re` vs `re_v2+all`

这是最关键的一组直接对比。

#### `lstm`

- `re`：`0.09129`
- `re_v2+all`：`0.08805`
- 差值：`-0.00324`

#### `transformer`

- `re`：`0.09884`
- `re_v2+all`：`0.09744`
- 差值：`-0.00140`

两条 backbone 上，`re_v2+all` 都没赢过 `re`。

但它的 `all_rankic_std` 都更低，这意味着：

> 当前的 V2 全奖励更像是在用更强的正则和 bonus 换取更平滑的跨年表现，但这一步还没有转化成更高的主要效果指标。

## 7. 哪些设计目前是合理的，哪些还不够合理

### 7.1 目前最合理的部分：结构奖励

不管看 V1 还是 V2，结构相关奖励都表现不错：

- `re_v2+struct × transformer` 当前第一；
- `re+struct × transformer` 也处于第一梯队；
- `re+struct × lstm` 也优于 `re × lstm`。

这说明一个结论：

> 在公式搜索里，结构探索本身确实有价值，问题不在“要不要结构项”，而在“结构项如何定义得更贴近价值”。

当前 V2 的“价值引导结构 bonus”方向是正确的。

### 7.2 当前证据不足的部分：功能增量奖励

`RI_func_v2` 的理论方向是对的，因为“残差化后的信息量”明显比“低相关惩罚”更合理。

但实验上它目前没有转化成明显收益：

- `re_v2+func` 不在前列；
- 其表现也没有稳定超过 `re+func`。

这说明问题可能不在思想，而在实现细节：

1. 使用 top-k pool 因子做拟合是否过于粗糙；
2. `ri_func_sample_size` 是否过小；
3. 用 `IC` 还是 `RankIC` 是否影响很大；
4. 当前线性残差化是否过弱。

结论是：

> `RI_func_v2` 现在不能下“无效”结论，但也没有足够证据证明它已经工作良好。

### 7.3 当前效果混合的部分：复杂度正则

`RI_reg` 的现象比较混合：

- `re_v2+reg × lstm` 很强；
- `re_v2+reg × transformer` 排名靠后；
- `re+reg × transformer` 反而较强；
- `re+reg × lstm` 排名最后。

这说明复杂度惩罚不是普适增益项，而更像：

> 在某些 backbone / 搜索分布下能抑制过拟合，在另一些设置下会压掉有用表达能力。

因此它应该保留，但不应该直接当成“默认必开项”。

### 7.4 当前不合理的结论方式：直接宣布 V2 全面胜出

基于当前结果，不能说：

> V2 已经全面优于 V1。

更准确的说法应该是：

1. V2 结构奖励方向值得继续推进；
2. V2 全量奖励还没赢；
3. V2 当前主要表现为“更稳一些，但不更强”。

## 8. 对论文/技术报告可直接使用的论证框架

### 8.1 奖励设计论证

可以这样写：

1. `RE` 是与任务目标直接对齐的主奖励，度量新因子加入后对组合表现的真实边际贡献。
2. 仅依赖 `RE` 会导致搜索易陷入局部模式，因此引入 RI 类项作为探索与正则辅助。
3. `RI_func` 应刻画条件剩余信息，而不是简单相关性惩罚。
4. `RI_struct` 应刻画“高价值但欠探索”的结构区域，而不是“越新越好”。
5. `RI_reg` 应控制表达式复杂度与风险，但不能过强压制有效结构。

### 8.2 实验论证

可以这样落：

1. 用 `post_rankic_mean` 衡量训练后时期真实泛化能力；
2. 用 `all_rankic_std` 或 `post_rankic_std` 衡量稳定性；
3. 用 `generalization_gap` 辅助分析时变漂移；
4. 比较 `reward_mode × backbone`；
5. 再做 `V1 vs V2` 与 `re vs re_v2+all` 的成对对比。

### 8.3 当前最稳妥的报告结论

当前最稳妥的写法不是“V2 全面提升”，而是：

> 实验表明，奖励设计中以真实边际收益 `RE` 作为主项是必要的；结构探索奖励在多个设置下表现出稳定增益，说明结构层面的价值引导探索具有实证合理性；相比之下，基于条件剩余信息的功能奖励与增强型复杂度正则尚未在当前实现中稳定转化为更高的训练后 RankIC，因此完整的 `re_v2+all` 组合虽提高了稳定性，但尚未整体超越基线 `re`。这表明 V2 的设计方向具有方法学合理性，但其不同子项的实现质量仍需进一步验证和调参。

## 9. 当前最需要补的实验

为了让论文论证更硬，建议补下面四组实验。

### 9.1 `re_v2` 纯 RE 对照

当前你有：

- `re`
- `re_v2+func`
- `re_v2+struct`
- `re_v2+reg`
- `re_v2+all`

但缺一个最关键的：

- `re_v2`

这个实验能回答：

> V2 的调度与框架本身是否就改变了训练动态？

没有这个对照，当前无法完全分离“V2 子项效果”和“V2 框架副作用”。

### 9.2 去信息泄漏的 checkpoint 选择

当前 `best_step_by_post_rankic` 用于分析没有问题，但不能作为主结果。

建议补两种口径：

1. `final_step`
2. `best_step_by_valid`

然后重新比较 `post_rankic_mean`。

### 9.3 `RI_func_v2` 参数敏感性实验

最需要扫的参数：

- `ri_func_metric`: `ic` vs `rankic`
- `ri_func_topk`
- `ri_func_sample_size`

如果这组实验不补，`RI_func_v2` 当前很难做强结论。

### 9.4 结构奖励强度扫描

建议补：

- `ri_struct_value_bonus`
- `ri_struct_underexplore_power`

因为当前证据最支持结构项，下一步最值得在这里深挖。

## 10. 下一阶段建议

如果目标是尽快形成一版能写进报告的稳定主结论，建议优先级如下：

1. 以 `transformer` 作为主 backbone。
2. 重点补 `re_v2`、`re_v2+struct`、`re_v2+all` 三组。
3. 用去泄漏的 step 选择方式重跑主表。
4. 把 `RI_func_v2` 先当作待验证项，而不是已经验证成功的创新点。

当前阶段最值得强调的创新点，应优先放在：

- “以边际收益 `RE` 为核心的奖励框架”
- “从纯新颖性转向价值引导结构探索”

而不是过早把 `RI_func_v2` 或 `re_v2+all` 宣称为稳定有效结论。

## 11. 相关结果文件

本报告对应的关键结果文件位于：

- [generalization_overview_runs.csv](C:/Users/qdz/Desktop/开题报告/alphagen/data/visualization_20260401/generalization_overview_runs.csv)
- [leaderboard_reward_backbone.csv](C:/Users/qdz/Desktop/开题报告/alphagen/data/visualization_20260401/leaderboard_reward_backbone.csv)
- [v1_vs_v2_by_backbone.csv](C:/Users/qdz/Desktop/开题报告/alphagen/data/visualization_20260401/v1_vs_v2_by_backbone.csv)
- [re_vs_re_v2all_by_backbone.csv](C:/Users/qdz/Desktop/开题报告/alphagen/data/visualization_20260401/re_vs_re_v2all_by_backbone.csv)

