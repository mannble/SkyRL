# SkyRL Harbor Meta Pipeline 当前流程审核文档

本文档描述当前代码里的 Harbor + Terminus-2 meta-learning pipeline。目标是让你在重新从头训练前，能逐项审核：哪些数据进入 meta，meta 如何生成策略和 Hook，哪些补丁会被金丝雀评估，哪些会最终进入 active patch library。

> 当前状态重点：`context_length` / `context_overload` 诊断已经在进入 planner 前被过滤，不会再驱动生成新的 context-window 类补丁。

---

## 1. 关键文件

| 模块 | 文件 | 作用 |
|---|---|---|
| 训练入口 | `examples/train_integrations/harbor/run_megatron.sh` | Megatron 训练脚本，覆盖 meta 参数 |
| meta 主控 | `examples/train_integrations/harbor/harbor_generator.py` | 采集轨迹、触发诊断、规划、剪枝、金丝雀、promote、生成 meta-RL 样本 |
| LLM 诊断器 | `skyrl-agent/skyrl_agent/meta_toolkit/diagnosis/interactive_diagnoser.py` | 多 worker 诊断失败轨迹 |
| 诊断工具 | `skyrl-agent/skyrl_agent/meta_toolkit/diagnosis/diagnosis_tools.py` | 给诊断 worker 用的 trace inspection 工具 |
| 规则诊断器 | `skyrl-agent/skyrl_agent/meta_toolkit/diagnosis/rule_based_diagnoser.py` | 没有 LLM 诊断时的 fallback |
| LLM planner | `skyrl-agent/skyrl_agent/meta_toolkit/editing/llm_patch_planner.py` | 生成策略补丁和 Hook 补丁 |
| rule planner | `skyrl-agent/skyrl_agent/meta_toolkit/editing/patch_planner.py` | 没有 LLM planner 时的规则补丁规划 |
| patch pruner | `skyrl-agent/skyrl_agent/meta_toolkit/editing/patch_pruner.py` | 策略/Hook 去重、合并、预算裁剪 |
| canary runner | `skyrl-agent/skyrl_agent/meta_toolkit/validation/canary_runner.py` | 运行 baseline/candidate 小规模评估 |
| canary comparison | `skyrl-agent/skyrl_agent/meta_toolkit/validation/comparison.py` | 计算 delta 和 regression |
| promoter | `skyrl-agent/skyrl_agent/meta_toolkit/runtime/promoter.py` | 决定是否接受候选补丁 |
| Hook runtime | `skyrl-agent/skyrl_agent/meta_toolkit/hooks/*.py` | HookContext、HookExecutor、静态校验、smoke test、kv 隔离 |
| Terminus-2 runtime | `/home/ray/dependencies/harbor/src/harbor/agents/terminus_2/terminus_2.py` | 运行时策略动态注入和 Hook 执行 |

---

## 2. 当前训练脚本里的 meta 参数

`run_megatron.sh` 当前覆盖了这些 meta 参数：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `generator.meta.enabled` | `true` | 开启 meta loop |
| `generator.meta.interval_batches` | `2` | 每 2 个 generator batch 触发一次 meta cycle |
| `generator.meta.max_candidates` | `2` | 每轮最多生成 2 个 candidate |
| `generator.meta.canary_num_tasks` | `16` | 每个 candidate 的 canary 任务数 |
| `generator.meta.canary_n_samples` | `4` | 每个 canary 任务每侧采样数 |
| `generator.meta.override_base` | `/home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus2` | active patch library 目录 |
| `generator.meta.llm_model` | `${META_LLM_MODEL:-$SERVED_MODEL_NAME}` | meta 诊断/planner/pruner 用的模型 |
| `generator.meta.llm_base_url` | `http://127.0.0.1:8000/v1` 默认 | meta LLM API base |
| `generator.meta.llm_temperature` | `0.7` | MetaLLMConfig 默认温度；子流程会按需要覆盖 |
| `generator.meta.llm_max_tokens` | `4096` | meta LLM 默认输出上限 |

`MetaLoopConfig` 里还有这些默认值没有在脚本里覆盖：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `diagnosis_num_workers` | `16` | 每轮最多启动 16 个诊断 worker |
| `diagnosis_partial_tasks` | `8` | partial 任务最多 8 个 |
| `diagnosis_all_fail_tasks` | `8` | all-fail 任务最多 8 个 |
| `diagnosis_all_pass_tasks` | `0` | 默认不诊断全成功任务 |
| `max_strategies` | `32` | 策略库硬上限 |
| `max_hook_groups` | `8` | Hook 组硬上限 |
| `canary_diagnosed_ratio` | `0.75` | canary 失败任务中倾向选择诊断命中的任务 |
| `canary_min_undiagnosed` | `2` | canary 中至少保留一些未诊断失败任务 |

---

## 3. 总体流程

```text
Harbor rollout
  -> 收集 Trial 轨迹 TraceRecord
  -> 每 interval_batches 触发 meta cycle
  -> 持久化轨迹
  -> LLM/规则诊断
  -> 过滤 context_length/context_overload
  -> LLM/规则 planner 生成 candidates
  -> 每个 candidate 复制 active patch library 到临时目录
  -> 在临时目录 apply candidate
  -> PatchPruner 对临时目录整体去重/控量
  -> 若 candidate 与 active patch library 内容完全相同，跳过 canary
  -> baseline 和 candidates 并行 canary
  -> 对比 reward delta 和 regression
  -> 只 promote 最好的 accepted candidate
  -> 注入 active overrides 到后续 rollout
  -> 记录 planning conversations 作为 meta-RL 样本
```

---

## 4. Active patch library 的加载和“从头训练”含义

Active patch library 默认在：

```text
/home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus2
```

这个目录里的内容会在 meta controller 初始化时被 `_preload_active_overrides()` 读入 `_ACTIVE_META_OVERRIDES`：

- `strategy_library.yaml` 会作为已有策略库加载。
- `hooks/*.py` 会作为已有 Hook 加载。
- baseline canary 会使用这些 active patches。
- 主训练 rollout 也会通过 `_inject_meta_overrides()` 使用这些 active patches。

所以如果你想“从头重新跑训练，不用已有补丁”，必须保证进程启动时 `override_base` 指向一个空 patch library，或者该目录下没有 `strategy_library.yaml` 和 `hooks/*.py`。本文档没有删除任何已有补丁。

---

## 5. 轨迹采集与 failure tags

每个 Harbor trial 会被转成 `TraceRecord`，meta 主要使用：

| 字段 | 用途 |
|---|---|
| `task_id` | 例如 `749-traj3`，用于诊断和 canary 任务匹配 |
| `metadata.prompt` | 原始 task path，用于 canary 重新运行同一个任务 |
| `success` / `final_reward` | 判断成功/失败和 canary reward |
| `finish_reason` | `complete`、`agent_timeout`、`context_length`、`error` 等 |
| `failure_tags` | `verification_failure`、`command_parsing_failure` 等标签 |
| `events` | 轨迹中的 LLM/terminal 事件，供诊断 worker 检查 |
| `turn_count` / `tool_calls` / `tool_failures` | 诊断统计和规则判断 |

当前 `context_length` 仍然可能被 trace bridge 标成 `context_overload`，但 meta 主控在诊断后会过滤：

```python
_IGNORED_PATCH_PROBLEM_TYPES = {"context_overload", "context_length"}
```

因此这两类 diagnosis 不会进入 planner，不会再驱动生成新补丁。

---

## 6. 诊断器流程

### 6.1 LLM 模式和规则模式

如果 `generator.meta.llm_model` 非空，使用 `InteractiveDiagnoser`。

如果 `llm_model` 为空，使用 `RuleBasedDiagnoser`。

当前 `run_megatron.sh` 会设置：

```bash
generator.meta.llm_model=${META_LLM_MODEL:-$SERVED_MODEL_NAME}
```

所以正常训练会走 LLM 诊断器。

### 6.2 任务分组

诊断器先按 task instance 聚合轨迹，例如：

```text
749-traj0
749-traj1
...
```

会聚合成 task key：

```text
749
```

然后分成三类：

| 类别 | 定义 | 当前上限 |
|---|---|---:|
| `partial` | 同一个任务有成功也有失败 | 8 |
| `all_fail` | 同一个任务全部失败 | 8 |
| `all_pass` | 同一个任务全部成功 | 0 |

选择 worker 时：

1. 先按上限选择 partial。
2. 再按上限选择 all_fail。
3. all_pass 默认上限是 0，所以不会选。
4. 如果 partial 或 all_fail 不足，会允许另一个失败桶补齐 worker 数量。
5. 不会为了凑 16 个 worker 强行加入 all-pass，除非你显式把 `diagnosis_all_pass_tasks` 设大于 0。

### 6.3 每个 worker 看到什么

每个 worker 只负责一个 task group。它会看到：

- 该 task group 的轨迹概览。
- 诊断历史摘要。
- 可调用的诊断工具说明。

它不会再看到整个 batch 的全局统计摘要；这样可以减少重复 token，也避免 worker 被其他任务的全局失败模式带偏。

当前暴露给 worker 的工具只有 4 个：

| 工具 | 作用 |
|---|---|
| `inspect_trace(task_id)` | 查看某条轨迹的详细事件 |
| `compare_traces(task_id_a, task_id_b)` | 对比两条轨迹 |
| `get_task_overview(task_key)` | 查看一个 task group 下所有轨迹 |
| `submit_diagnosis(diagnoses)` | 提交最终诊断 |

### 6.4 diagnosis 输出协议

`partial` worker 可以输出 `strategy_suggestions`，因为它能对比同任务的成功/失败轨迹。

```json
{
  "tool": "submit_diagnosis",
  "diagnoses": [
    {
      "problem_type": "verification_failure",
      "root_cause_hypotheses": ["..."],
      "confidence": 0.8,
      "affected_trace_ids": ["749-traj3"],
      "strategy_suggestions": [
        {
          "pattern": "...",
          "steps": ["..."],
          "source": "749-traj0 succeeded by ..."
        }
      ]
    }
  ]
}
```

`all_fail` 和 `all_pass` worker 不应该输出 `strategy_suggestions`。代码里也有 hard guard：只有 `task_category == "partial"` 的建议会被保留。

### 6.5 diagnosis 合并

所有 worker 并行执行，之后按 `problem_type` 合并：

- 同类 `problem_type` 会合并成一个 diagnosis 条目，避免 planner 看到大量重复类型。
- 合并条目会保留该类型下所有 worker 的 hypotheses、affected_trace_ids、analysis_summary 和 per-worker evidence。
- `strategy_suggestions` 只保留 partial worker 的输出；all-fail/all-pass worker 的 strategy_suggestions 会被过滤。
- 合并后的 `confidence` 取该 problem_type 下所有 worker confidence 的最大值，不做平均，也不因为 worker 数量多而额外抬高。
- metadata 里会记录 `confidence_policy=max_worker_confidence`、`merged_confidences` 和 `worker_evidence`，方便审计每个 worker 的原始诊断信息。
- strategy_suggestions 只保留 partial worker 的。

### 6.6 context 诊断过滤

诊断完成后，进入 planner 前会执行过滤：

```text
drop problem_type in {"context_overload", "context_length"}
```

这意味着：

- 诊断 worker 仍可能提到 context 问题。
- fallback 也仍可能合成 `context_overload` diagnosis。
- 但这些 diagnosis 会在 planner 前被丢弃。
- 当前 cycle 的 planner、pruner、canary 不会因为 context_length 生成候选补丁。

---

## 7. Planner 流程

### 7.1 Planner 看到什么

当前 LLM planner 看到的是诊断信息，不看完整 sample traces。

代码里 `include_traces=False`，所以 strategy planner 和 hook planner 都主要基于：

- problem_type
- hypotheses
- affected_trace_ids
- diagnosis analysis summary
- partial worker 提供的 strategy_suggestions

策略 planner 会看到 `strategy_suggestions`。

Hook planner 不会看到 `strategy_suggestions`，只看 diagnosis。

### 7.2 每个 candidate 怎么生成

每个 candidate group 会并行启动：

| 子 agent | 生成内容 |
|---|---|
| strategy sub-agent | 只生成 `strategy_edits` |
| hook sub-agent: `prompt_guidance` | 只允许生成 `before_llm_call` |
| hook sub-agent: `runtime_recovery` | 允许生成 `after_execute`、`on_timeout`、`after_round` |

当前 Hook category：

```python
_HOOK_CATEGORIES = {
    "prompt_guidance": ("before_llm_call",),
    "runtime_recovery": ("after_execute", "on_timeout", "after_round"),
}
```

每个 candidate group 会把 strategy partial 和 hook partial 合并成一个 `PatchCandidate`。

`max_candidates=2` 时，每轮最多有 2 个 candidate group。

### 7.3 策略生成协议

策略 planner 当前只允许 `add`：

```json
{
  "strategy_edits": [
    {
      "action": "add",
      "strategy": {
        "pattern": "...",
        "steps": ["..."],
        "scope": "general",
        "triggers": ["..."],
        "anti_triggers": ["..."]
      }
    }
  ],
  "rationale": "..."
}
```

它不能编辑或删除已有策略；去重、合并、删除都交给后面的 `PatchPruner`。

### 7.4 策略 metadata

策略可以带这些选择性注入 metadata：

| 字段 | 含义 |
|---|---|
| `scope` | `core`、`general`、`task_specific` |
| `triggers` | 任务指令中出现这些词时更可能注入 |
| `anti_triggers` | 任务指令中出现这些词时不注入 |
| `priority` | 可选，影响排序 |

### 7.5 Hook 生成协议

Hook planner 输出：

```json
{
  "code_hooks": {
    "after_round": "def hook(...): ..."
  },
  "rationale": "..."
}
```

当前允许生成的 Hook：

| Hook | 时机 | 返回语义 |
|---|---|---|
| `before_llm_call` | 每次 LLM call 前 | 返回 `{"append_prompt": "..."}`，只追加短提示 |
| `after_execute` | 命令成功执行后 | observer，只写 `context.kv`，返回值被忽略 |
| `on_timeout` | 命令 timeout 后 | observer，只写 `context.kv`，返回值被忽略 |
| `after_round` | base observation 构造完成后 | 返回 `{"next_prompt": "..."}`，只追加提示 |

`before_execute` 不再由 planner 生成，也不会在 Terminus-2 主循环里执行 learned hook。`PatchPruner` 在候选预评估阶段还会删除 `before_execute` 文件。

### 7.6 Hook repair 和 smoke test

Hook 生成后会进入 `_repair_hooks()`：

1. 静态 AST 校验：
   - 必须有且只有一个 `hook` 函数。
   - import 只能来自允许模块。
   - 未定义变量会报错。
   - `context.kv["key"]` 裸读会被拒绝，要求 `get` / `setdefault` / 默认赋值。
2. 编译 hook。
3. smoke test 多场景运行：
   - `before_llm_call`: initial / middle / periodic
   - `after_execute`: empty / cat / ls / missing path / permission / large output
   - `on_timeout`: simple / broad search / install / cat loop
   - `after_round`: empty / complete / cat / ls_complete / missing_path / timeout / 多轮状态
4. 如果失败，最多让 LLM 修复 3 次。

---

## 8. Candidate 临时目录与 PatchPruner

### 8.1 临时目录在哪里

每个 candidate 会创建：

```text
/tmp/meta_canary_candidate_<N>_*
```

流程：

1. 把当前 `override_base` 整体复制到临时目录。
2. 把 candidate 的 strategy/hook patch apply 到临时目录。
3. 在临时目录里运行 `PatchPruner`。
4. 使用临时目录内容做 canary。
5. 结束后清理临时目录。

也就是说，candidate 不是直接写 active patch library，而是先在副本里评估。

### 8.2 无变化 candidate 会跳过 canary

prune 后会计算临时目录 patch content signature。

如果 candidate 应用和剪枝后，与当前 active patch library 完全相同：

- 不跑 canary。
- 记为 `noop_skip`。
- candidate delta 设为 `-0.05`。

---

## 9. Strategy Pruner

`PatchPruner` 在 canary 前对整个策略库做治理：已有策略 + 新 candidate 策略一起看。

### 9.1 第一阶段：LLM curate

如果策略数大于 1，会把所有策略按 `[0] ... [N]` 编号后给 LLM。

LLM 只能返回：

```json
{
  "delete_indices": [1, 3],
  "merge_groups": [
    {
      "replace": 0,
      "drop": [2, 5],
      "merged_strategy": {
        "pattern": "...",
        "steps": ["..."],
        "scope": "general",
        "triggers": ["..."],
        "anti_triggers": ["..."]
      },
      "reason": "..."
    }
  ],
  "notes": ["..."]
}
```

语义：

- `delete_indices`: 删除 standalone 低质量策略。
- `merge_groups`: 合并重复或冲突策略。
- `replace`: 用合并后的策略覆写哪个现有编号。
- `drop`: 删除同组里被合并掉的编号。
- `merged_strategy`: 只能合并 `replace + drop` 中已经存在的思想，不能发明新策略。

### 9.2 第二阶段：budget delete

如果 LLM curate 后策略仍超过 `max_strategies=32`：

1. 再调用一次 LLM，要求只输出应该删除哪些策略。
2. 这一步不允许 merge，只允许 delete。
3. 如果 LLM 删除不够，规则兜底会按冗余度删除最重复的策略，直到不超过 32。

---

## 10. Hook Pruner

Hook pruner 也是在 canary 前对整个 Hook library 治理。

### 10.1 Hook group

Hook 数量限制是按“Hook 组”算，而不是按 hook point。

同一个 planner 子 agent 生成的一组 Hook 会共享 group key，例如：

```text
candidate_0_hookgrp_runtime_recovery
```

这个组里可能同时有：

- `after_execute`
- `on_timeout`
- `after_round`

这样 observer hook 可以往 `context.kv` 写信号，`after_round` 再消费这个信号。

### 10.2 第一阶段：基础清理

Pruner 会先做：

- 删除 disabled hook point：`before_execute`
- 删除 exact duplicate hooks
- 构建 hook groups

### 10.3 第二阶段：LLM 删除重复/低质组

LLM 看到所有 Hook group，编号 `[0] ... [N]`，每组显示完整 Hook 源码。

LLM 只返回：

```json
{
  "delete_groups": [0, 3],
  "notes": ["..."]
}
```

它不能改 Hook 代码，只能删除整组。

### 10.4 第三阶段：Hook 组预算

如果 Hook group 数量仍超过 `max_hook_groups=8`：

1. 再调用 LLM，只要求删到预算以内。
2. 如果 LLM 删除不够，规则兜底按相似度/冗余度继续删除。
3. 最终保证 group 数量不超过 8。

---

## 11. Canary 任务选择

Canary 从当前 meta cycle 的 traces 中选任务。

当前训练脚本：

```text
canary_num_tasks = 16
canary_n_samples = 4
```

每个 candidate 会跑：

```text
16 tasks × 4 samples = 64 trials
```

baseline 也跑同样数量。

### 11.1 任务桶

Canary 会按 task path 聚合当前 traces：

| 桶 | 定义 |
|---|---|
| diagnosed_failed | 失败任务，且 diagnosis 的 affected_trace_ids 命中过 |
| undiagnosed_failed | 失败任务，但没被 diagnosis 命中过 |
| success_tasks | 当前 traces 中没有失败、有成功样本的任务 |

### 11.2 选择策略

1. 最多选 `canary_num_tasks` 个 task。
2. 最多保留 1/4 中的 1 个成功任务做 regression check；当前 16 个任务时最多 1 个。
3. 剩余 slot 主要给失败任务。
4. 失败任务中优先诊断命中的任务，但保留至少 `canary_min_undiagnosed=2` 个未诊断失败任务。
5. 如果某一类不足，会从其他失败任务或成功任务 backfill。

---

## 12. Canary 执行和接受规则

### 12.1 baseline 与 candidates 并行

当前流程会并行运行：

- baseline：使用当前 active patch library。
- candidate：使用 candidate 临时目录中的 merged + pruned patch library。

每个 task 的多个 sample reward 会先平均成 per-task avg reward。

### 12.2 比较指标

`ComparisonEngine` 计算：

```text
delta_score = mean(candidate_task_rewards) - mean(baseline_task_rewards)
```

Regression 判断：

| 条件 | 结果 |
|---|---|
| `delta < -0.05` | regression |
| 任一 task reward 下降超过 `0.3` | regression |
| task 数不足 | regression |

### 12.3 Promote 规则

`Promoter` 接受 candidate 的条件：

1. canary 没有 error。
2. 没有 regression。
3. `delta_score >= 0.01`。

如果多个 candidate 都通过，只 promote `delta_score` 最大的那个。

Promote 时：

1. restore 到本轮开始前的 active patch snapshot。
2. 把 best candidate 的临时目录同步回真实 `override_base`。
3. 写 accepted patch 审计记录。
4. 在 patch version-control repo 中 commit。
5. 把 best candidate overrides 注入 `_ACTIVE_META_OVERRIDES`，后续 rollout 立即使用。

---

## 13. 运行时策略注入

Terminus-2 在每个任务开始时调用 `_build_strategy_injection()`。

它不会把所有策略都注入，而是动态选择：

| 类型 | 数量上限 | 选择方式 |
|---|---:|---|
| `core` | 5 | 排序后取前 5 |
| `general` | 3 | 必须和当前 task instruction 有 trigger/token 匹配 |
| `task_specific` | 2 | 必须有更强匹配，通常需要具体 artifact/path 命中 |

总注入字符上限：

```text
_MAX_STRATEGY_INJECTION_CHARS = 10000
```

其他保护：

- `anti_triggers` 命中则不注入。
- 没写 `scope` 时会自动推断。
- 含硬编码危险片段的策略会被跳过。
- 注入格式是 `[LEARNED STRATEGIES]`，追加在原任务说明后面。

---

## 14. 运行时 Hook 执行

### 14.1 HookContext

Hook 只能拿到 `HookContext`，不能拿到 agent 内部对象。除了 `context.kv`，其他字段语义上都是只读。

常用字段：

| 字段 | 含义 |
|---|---|
| `episode` | 当前轮次，0-based |
| `total_episodes` | 配置的安全上限 |
| `last_commands` | 最近一轮解析出的 command strings |
| `last_analysis` / `last_plan` | 最近一轮解析出的 analysis / plan |
| `is_task_complete` | 最近一轮解析出的 completion flag |
| `last_terminal_output` | 最近一轮真实 terminal output |
| `next_observation` | base observation，即不加 Hook guidance 时下一轮会看到的内容 |
| `execution_history` | 最近最多 20 轮历史 |
| `kv` | 唯一可写的跨轮状态 |

### 14.2 Hook 安全语义

| Hook | 当前 runtime 行为 |
|---|---|
| `before_llm_call` | 可以返回 `append_prompt`，runtime 只追加，不允许重写 prompt |
| `after_execute` | 只作为 observer，返回值被忽略，不允许改 terminal output |
| `on_timeout` | 只作为 observer，返回值被忽略，不允许改 timeout output |
| `after_round` | 可以返回 `next_prompt`，runtime 只追加到 base observation 后面 |
| `before_execute` | learned hook 不执行 |

追加文本上限：

```text
_MAX_HOOK_APPEND_CHARS = 1200
```

### 14.3 多 Hook 运行和 kv 隔离

HookExecutor 支持同一 hook point 多个 Hook。

- `before_llm_call`：多个 Hook 的 `append_prompt` 会拼接。
- `after_execute` / `on_timeout`：每个 Hook 都独立观察原始输出，返回值忽略。
- `after_round`：每个 Hook 都看同一份原始输入，多个 `next_prompt` 会拼接。

`context.kv` 按 Hook group 隔离：

- 同一个 generated hook group 内的 sibling hooks 共享 kv。
- 不同 group 的 hooks 使用不同 kv namespace。
- 这样 `after_execute` 可以写信号，组内 `after_round` 可以读，但不会污染其他 Hook 组。

---

## 15. 日志和审计产物

每次 meta run 会创建：

```text
{log_dir}/meta_run_<timestamp>/
```

主要文件：

| 文件 | 内容 |
|---|---|
| `events.jsonl` | diagnosis/planner/canary/pruner 的结构化事件，按 `stream` 区分 |
| `llm_conversations.jsonl` | diagnosis/planning LLM 对话 |
| `cycle_summary.md` | 人类可读的每轮总结 |
| `diagnosis_history.jsonl` | 跨 cycle 的历史摘要 |
| `accepted_patches/*.json` | 被接受 candidate 的审计记录 |

---

## 16. Meta-RL 样本

每轮 meta cycle 结束后，会从 LLM client 里取出对话记录。

当前只把 `planning` role 的 conversations 变成 `MetaTrainingSample`：

- 同一个 cycle 的 planning samples 共享一个 `instance_id`。
- 每个 candidate 对应一个 `repetition_id`。
- reward 使用该 candidate 的 canary delta。
- 如果 candidate prepare/canary 失败，reward 会是负值或 `-1.0`。
- 诊断 conversations 会保存日志，但当前不作为 meta-RL 样本训练。

这意味着训练主要在学：

```text
什么样的 planner 输出能产生 canary delta 更高的补丁
```

---

## 17. 从头训练前建议审核项

| 项 | 当前结论 |
|---|---|
| 是否还会生成 context_length/context_overload 补丁 | 正常不会，已在 planner 前过滤 |
| 现有补丁是否会被自动加载 | 会，只要 `override_base` 下还有 `strategy_library.yaml` 或 `hooks/*.py` |
| 是否会注入所有策略 | 不会，运行时 5 core + 3 general + 2 task_specific，最多 10000 字符 |
| 策略是否会越积越多 | pruner 硬上限 32 |
| Hook 是否会越积越多 | pruner 按 group 硬上限 8 |
| Hook 能否改命令 | learned `before_execute` 不执行；命令只走 runtime validator |
| Hook 能否改 terminal output | `after_execute` / `on_timeout` 返回值忽略 |
| after_round 能否新增轮次 | 不能直接新增；只能把 `next_prompt` 追加到下一轮 observation |
| promote 是否接受所有正收益候选 | 不会，只 promote 通过阈值后 delta 最大的一个 |
| candidate 没实际改动是否还跑 canary | 不会，signature 相同会 `noop_skip` |

---

## 18. 重新从空补丁训练的操作含义

如果要从空 patch library 开始，启动训练前需要满足：

```text
override_base/
  strategy_library.yaml    不存在或为空
  hooks/*.py               不存在
```

或者把 `generator.meta.override_base` 指向一个新的空目录。

否则当前 `_preload_active_overrides()` 会把已有补丁预加载，导致：

- 主 rollout 已经带旧策略/Hook。
- canary baseline 也带旧策略/Hook。
- 新 candidate 是在旧补丁基础上继续叠加和剪枝。

本文档没有执行删除或移动补丁，只描述当前机制。
