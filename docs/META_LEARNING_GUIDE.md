# 元学习闭环指南

## 概述

本指南描述了 Harbor + SkyRL 训练的元学习闭环实现。系统会自动诊断 agent 失败原因，生成两类补丁（策略库 + 代码钩子），在金丝雀任务集上评估，并提升成功的补丁。

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     HarborGenerator.generate()                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  1. 运行 Harbor Trials → 收集 TraceRecords               │  │
│  │  2. 每 N 个 batch：                                       │  │
│  │     ├─ 持久化 traces 到 JSONL                            │  │
│  │     ├─ 诊断失败 (InteractiveDiagnoser / RuleBasedDiagnoser)│  │
│  │     ├─ 规划补丁 (LLMPatchPlanner)                        │  │
│  │     │   ├─ Tier 1: strategy_edits (策略库编辑)           │  │
│  │     │   └─ Tier 2: code_hooks (代码钩子，含 after_round) │  │
│  │     ├─ 执行补丁 (PatchExecutor → YAML + Python hooks)   │  │
│  │     ├─ 金丝雀评估 (CanaryRunner, 并发)                   │  │
│  │     └─ 如果 delta > 0 则提升 (Promoter)                 │  │
│  │  3. 将策略库 + 代码钩子注入到下一批 Harbor trials        │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 两层修改机制

**Tier 1: 策略库 (strategy_library)**
- 提示词级别的修改：向 agent 的指令中注入 `[LEARNED STRATEGIES]` 段
- 支持 add/edit/remove 操作
- 低风险，即时生效

**Tier 2: 代码钩子 (code_hooks)**
- 代码级别的修改：Python 函数注入到 agent 执行循环的特定位置
- 6 个钩子点: before_llm_call, before_execute, after_execute, on_timeout, on_parse_error, **after_round**
- `after_round` 是最强大的钩子：可以触发额外的 LLM 调用、拒绝完成、强制退出、注入文本
- 高影响力，所有钩子在沙盒中运行，失败时回退到原始值

## 补丁版本管理 (PatchVersionControl)

每个被接受的 agent 补丁都会自动提交到 `override_base` 目录下的**独立 git 仓库**，与 SkyRL 主仓库完全隔离。

### 工作原理

1. **初始化**: `_MetaLoopController` 启动时在 `meta_patches/terminus2/` 内执行 `git init`（已有仓库则跳过）
2. **自动提交**: `Promoter` 每次 accept 补丁后自动 `git add -A && git commit`，commit message 包含：
   - `candidate_id`: 补丁标识
   - `cycle`: 元学习周期编号
   - `delta_score`: 金丝雀评估得分差
   - `timestamp`: 提交时间
3. **回滚**: 如果发现某个补丁导致 agent 性能下降，可回滚到之前的任意版本

### 查看补丁历史

```bash
cd /home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus2
git log --oneline
```

### 回滚到之前版本

```bash
cd /home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus2
git log --oneline          # 找到要回滚的 commit SHA
git checkout <SHA> -- .    # 恢复文件
git add -A && git commit -m "manual rollback to <SHA>"
```

### 编程式回滚

```python
from skyrl_agent.meta_toolkit.runtime import PatchVersionControl, PatchVersionControlConfig

vc = PatchVersionControl(PatchVersionControlConfig(
    override_base="/path/to/meta_patches/terminus2"
))
vc.rollback("HEAD~1")       # 回滚一个版本
vc.rollback("abc1234")      # 回滚到指定 SHA
history = vc.history(n=10)  # 查看最近 10 次提交
```

### 标记已知良好状态

```python
vc.tag("v1-baseline", message="Initial override state before training")
```

## 配置

### 启用元学习

在训练命令或 Hydra 配置中添加以下参数：

```bash
generator.meta.enabled=true                    # 启用元学习循环
generator.meta.interval_batches=20             # 每 N 个 batch 运行一次元学习周期
generator.meta.max_candidates=2                # 每次周期最多评估的补丁候选数
generator.meta.canary_num_tasks=4              # 金丝雀评估的任务数
generator.meta.override_base=/path/to/overrides  # Override 文件目录
generator.meta.log_dir=/tmp/skyrl-logs         # Trace 持久化目录

# Phase 3: LLM-based 诊断与补丁规划（可选）
generator.meta.llm_model=gpt-4o               # LLM 模型名称（空字符串 = 使用规则）
generator.meta.llm_base_url=http://localhost:8000/v1  # OpenAI-compatible API 地址
generator.meta.llm_api_key=""                  # API key（可通过 META_LLM_API_KEY 环境变量设置）
```

### 配置参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `enabled` | bool | `true` | 启用/禁用元学习闭环 |
| `interval_batches` | int | `20` | 每 N 个 generator batch 运行一次元学习周期 |
| `max_candidates` | int | `2` | 每次周期最多评估的补丁候选数 |
| `canary_num_tasks` | int | `4` | 金丝雀评估的任务数量 |
| `override_base` | str | `meta_patches/terminus2/` | Override YAML 文件目录 |
| `log_dir` | str | `/tmp/skyrl-logs` | Trace 持久化的根目录 |
| `llm_model` | str | `""` | 元推理 LLM 模型名。设置后启用 Phase 3 LLM 模式，留空则使用规则 |
| `llm_base_url` | str | `http://localhost:8000/v1` | OpenAI-compatible API 端点 |
| `llm_api_key` | str | `""` | API key。也可通过环境变量 `META_LLM_API_KEY` 或 `OPENAI_API_KEY` 设置 |
| `llm_temperature` | float | `0.3` | LLM 采样温度 |
| `llm_max_tokens` | int | `2048` | LLM 最大输出 token 数 |

### 示例：完整训练命令

```bash
bash /home/ray/SkyRL/examples/train_integrations/harbor/run_megatron.sh \
  generator.meta.enabled=true \
  generator.meta.interval_batches=20 \
  generator.meta.max_candidates=2 \
  generator.meta.canary_num_tasks=4
```

## 组件

### 1. Trace 持久化

**模块**: `skyrl_agent.meta_toolkit.observability.trace_writer`

将 `TraceRecord` 对象写入 JSONL 文件以供后续分析。

**输出位置**: `{log_dir}/{run_name}/meta_traces/traces.jsonl`

**TraceRecord 字段**:
- `task_id`: 唯一任务标识符
- `agent_version`: Agent 名称（例如 "terminus-2"）
- `success`: 布尔成功标志
- `final_reward`: 任务奖励（0.0 - 1.0）
- `turn_count`: Agent 回合数
- `finish_reason`: "complete" | "agent_timeout" | "error" | "context_length"
- `sync_bottleneck_score`: 0.0 - 1.0（1.0 = 严重超时）
- `failure_tags`: 失败类型列表（例如 ["sync_bottleneck", "recovery_failure"]）
- `tool_calls`: 总工具调用次数
- `tool_failures`: 失败的工具调用次数
- `events`: TraceEvent 对象列表

### 2. 诊断器

根据 `llm_model` 配置自动选择模式：

#### 规则模式（默认，`llm_model=""` 时）

**模块**: `skyrl_agent.meta_toolkit.diagnosis.rule_based_diagnoser`

分析一批 `TraceRecord` 对象并通过预定义规则识别失败模式。

**支持的问题类型**:
- `sync_bottleneck`: Agent 在长时间运行的任务上花费太多连续回合
- `timeout`: Agent 超出时间预算
- `verification_failure`: 任务完成但验证失败
- `recovery_failure`: 工具错误发生且 agent 未恢复
- `termination_failure`: Agent 过早放弃
- `planning_failure`: 规划器生成无效的子目标
- `context_overload`: 上下文窗口在任务完成前填满
- `tool_exhaustion`: 工具调用过多而未收敛
- `low_reward`: 一般性低奖励，无特定模式

#### LLM 模式（Phase 3，`llm_model` 非空时）

**模块**: `skyrl_agent.meta_toolkit.diagnosis.llm_diagnoser`

将 traces 序列化后发送给 LLM，由 LLM 分析失败模式并返回结构化的 `DiagnosisResult`。

**优势**:
- 可发现规则未覆盖的新型失败模式
- 假设更具体，引用具体的 task ID 和 trace 证据
- 可自定义问题类型，不限于预定义列表

**容错**: LLM 调用失败时自动降级为规则模式。

**输出**: `DiagnosisResult`
- `problem_type`: 问题类型（LLM 模式可自定义）
- `root_cause_hypotheses`: 人类可读的假设列表
- `candidate_modules`: 建议修补的模块（例如 ["planner_policy", "retry_policy"]）
- `confidence`: 0.0 - 1.0
- `affected_task_ids`: 匹配此问题的任务 ID 列表

### 3. 补丁规划器

根据 `llm_model` 配置自动选择模式：

#### 规则模式（默认）

**模块**: `skyrl_agent.meta_toolkit.editing.patch_planner`

从 `DiagnosisResult` + `ModuleRegistry` 通过关键词映射生成 `PatchCandidate` 对象。

#### LLM 模式（Phase 3）

**模块**: `skyrl_agent.meta_toolkit.editing.llm_patch_planner`

将诊断结果和失败 traces 发送给 LLM，由 LLM 决定具体的 override 字段和值。

**优势**:
- 可生成 `system_hint` 等自由文本字段（规则模式做不到）
- 可根据具体 trace 证据调整参数值（而非固定值）
- 可组合多个字段的变更（规则模式每个问题类型只有固定套路）

**容错**: LLM 调用失败时自动降级为规则模式。

**允许修补的模块**:
- `planner_policy`: 高层任务规划和子目标分解
- `retry_policy`: 工具或测试失败后的恢复
- `verification_policy`: 完成前或代码更改后的验证
- `finish_policy`: 终止决策和完成原因

**输出**: `PatchCandidate`
- `target_modules`: 要修补的模块名称列表
- `files`: `PatchFileEdit` 对象列表（路径、变更类型、意图）
- `risk`: "low" | "medium" | "high"
- `required_tests`: ["canary_eval"]
- `notes`: 人类可读的描述

### 4. 补丁执行器

**模块**: `skyrl_agent.meta_toolkit.editing.patch_executor`

通过写入 YAML override 文件来应用 `PatchCandidate` 编辑。

**Phase 1 保证**: 永远不会触碰 Harbor Python 源文件。只写入 SkyRL 控制的 override 目录。

**Override 文件位置**:
```
{override_base}/
├── planner_overrides.yaml
├── retry_overrides.yaml
├── verification_overrides.yaml
└── finish_overrides.yaml
```

**Override 字段**:

#### planner_overrides.yaml
```yaml
system_hint: ""                    # 额外的系统级指令
max_replans: 3                     # 最大重新规划迭代次数
subgoal_depth: 3                   # 首选的子目标数量
planning_style: "lazy"             # "eager"（立即分解）| "lazy"（延迟子目标）
max_tool_calls_per_plan: 10        # 重新规划前的工具调用硬上限
```

#### retry_overrides.yaml
```yaml
max_retries: 2                     # 最大重试次数
retry_delay_sec: 2.0               # 重试前的基础延迟
backoff_multiplier: 1.5            # 指数退避乘数
fallback_on_failure: true          # 最终失败时提供后备答案
```

#### verification_overrides.yaml
```yaml
verify_after_each_step: false      # 每次工具调用后运行验证
verification_threshold: 0.7        # 通过验证的最低置信度分数
skip_verification_tags: []         # 跳过验证的步骤标签
```

#### finish_overrides.yaml
```yaml
require_confirmation: true         # Agent 必须明确确认完成
min_steps_before_finish: 3         # 允许完成前的最少工具调用次数
early_exit_reward_threshold: 1.0   # 如果奖励超过此值则提前退出
```

### 5. 金丝雀评估

**模块**: `skyrl_agent.meta_toolkit.validation.canary_runner`

在提升前在小任务集上评估补丁候选。

**Phase 2 行为（当前）**:
- `_MetaLoopController` 在初始化时构建一个 trial runner 闭包（`_build_trial_fn`），该闭包可以独立运行 Harbor Trial 并返回 reward。
- 每次元学习周期触发时，从累积的 traces 中提取最多 `canary_num_tasks` 个任务路径（优先选择失败任务，补充少量成功任务用于回归检测）。
- 对每个补丁候选，`CanaryRunner` 串行执行两轮 Trial：
  1. **Baseline 轮**：使用当前激活的 `_ACTIVE_META_OVERRIDES`（如果没有则不注入 overrides）。
  2. **Candidate 轮**：使用新生成的候选 overrides。
- 两轮都对相同的 task 集运行，分别收集 reward 列表。
- `ComparisonEngine` 计算 `delta_score = avg(after) - avg(before)`，检测回归。
- 如果 `delta > 0` 且无回归，`Promoter` 接受补丁并激活新 overrides。

**任务选择策略**:
- 从近期失败 traces 中提取 task path（存储在 `trace.metadata["prompt"]` 中）
- 优先使用失败任务（它们是补丁的优化目标）
- 补充少量成功任务用于检测回归
- 去重后取最多 `canary_num_tasks` 个

**输出**: `CanaryRunResult`
- `candidate_id`: 唯一标识符
- `patch_eval_result`: 包含前后分数的 `PatchEvalResult`
- `before_scores`: 无补丁（baseline）的每任务 reward 列表
- `after_scores`: 有补丁（candidate）的每任务 reward 列表
- `task_ids`: 用于金丝雀的任务路径列表

**PatchEvalResult 字段**:
- `before_score`: baseline 平均分数
- `after_score`: candidate 平均分数
- `delta_score`: `after_score - before_score`
- `regression`: 布尔标志（如果 delta < -0.05 则为 true）
- `notes`: 人类可读的摘要

### 6. 提升器

**模块**: `skyrl_agent.meta_toolkit.runtime.promoter`

根据金丝雀评估决定是否接受或拒绝补丁候选。

**接受规则**:
1. `delta_score > min_delta`（默认：0.0）
2. `delta_score >= reject_threshold`（默认：-0.05）
3. 无金丝雀评估错误

**接受时**:
1. 将 `accepted_patch.json` 写入 `{accepted_patches_dir}/{candidate_id}.json`
2. 激活 override 文件（已由 PatchExecutor 写入）
3. 将 overrides 注入到下一批 Harbor trials

**拒绝时**:
1. 记录拒绝原因
2. Override 文件保留但不激活

**输出**: `PromotionDecision`
- `candidate_id`: 唯一标识符
- `decision`: "accept" | "reject"
- `delta_score`: 分数改进
ession`: 布尔标志
- `notes`: 人类可读的原因
- `promoted_at`: ISO 时间戳（如果接受）

### 7. Harbor Terminus2 集成

**修改的文件**: `/home/ray/dependencies/harbor/src/harbor/agents/terminus_2/terminus_2.py`

**新参数**:
- `meta_overrides`: Override 值的字典（由 HarborGenerator 注入）
- `meta_overrides_path`: 包含 overrides 的 YAML 文件路径（字典的替代方案）

**行为**:
1. 从 `meta_overrides_path` 加载 overrides（如果提供）
2. 与 `meta_overrides` 字典合并（如果提供）
3. 通过 `_build_meta_override_section()` 构建 override 部分
4. 将 override 部分前置到任务指令

**Override 部分格式**:
```
[SKYRL META OVERRIDES]
[PLANNING STYLE] eager
[MAX REPLANS] 3
[MAX RETRIES] 3
[VERIFY EVERY STEP] true
[MIN STEPS BEFORE FINISH] 5
```

此部分前置到任务指令，以便模型可以调整其行为。

## 工作流程

### 逐步执行

1. **训练开始**
   - HarborGenerator 以 `meta.enabled=true` 初始化
   - 使用 `ModuleRegistry` 创建 `_MetaLoopController`

2. **生成批次**
   - Harbor trials 使用当前激活的 overrides 运行
   - 每个 trial 产生一个 `TraceRecord`
   - Traces 在 `_MetaLoopController` 中累积

3. **元学习周期触发**（每 `interval_batches` 个批次）
   - **持久化**: 将累积的 traces 写入 `{log_dir}/{run_name}/meta_traces/traces.jsonl`
   - **诊断**: `LLMDiagnoser` 或 `RuleBasedDiagnoser` → `DiagnosisResult[]`
   - **规划**: `LLMPatchPlanner` 或 `PatchPlanner` → `PatchCandidate[]`（最多 `max_candidates`）
   - **执行**: `PatchExecutor.apply(candidate)` → 写入 override YAML 文件
   - **金丝雀**: `CanaryRunner.evaluate_candidate()` → `CanaryRunResult`（baseline + candidate 并发）
   - **提升**: `Promoter.promote()` → `PromotionDecision`
     - 如果接受：更新 `_ACTIVE_META_OVERRIDES` 注册表
     - 如果拒绝：丢弃候选

4. **下一批次**
   - HarborGenerator 将 `_ACTIVE_META_OVERRIDES` 注入到 trial 配置
   - Harbor Terminus2 读取 overrides 并调整行为
   - 周期重复

### 记录的指标

元学习指标合并到 `rollout_metrics`：

```python
{
    "meta/cycle_batch": 20,              # 周期运行时的批次号
    "meta/num_diagnoses": 2,             # DiagnosisResult 对象数量
    "meta/num_candidates": 2,            # PatchCandidate 对象数量
    "meta/promoted": 1,                  # 接受的补丁数量
    "meta/rejected": 1,                  # 拒绝的补丁数量
    "meta/num_trace_records": 128,       # 批次中的总 traces
    "meta/num_failure_tags": 3,          # 唯一失败标签
    "meta/num_candidate_modules": 2,     # 唯一候选模块
}
```

## 调试

### 检查 Trace 持久化

```bash
# 查看 traces
cat /tmp/skyrl-logs/{run_name}/meta_traces/traces.jsonl | jq .

# 计数 traces
wc -l /tmp/skyrl-logs/{run_name}/meta_traces/traces.jsonl
```

### 检查 Override 文件

```bash
# 查看激活的 overrides
cat /home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus2/planner_overrides.yaml
cat /home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus2/retry_overrides.yaml
cat /home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus2/verification_overrides.yaml
cat /home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus2/finish_overrides.yaml
```

### 检查已接受的补丁

```bash
# 查看已接受的补丁
ls /tmp/skyrl-logs/accepted_patches/
cat /tmp/skyrl-logs/accepted_patches/candidate_1.json | jq .
```

### 启用调试日志

添加到训练脚本：

```python
import logging
logging.getLogger("skyrl_agent.meta_toolkit").setLevel(logging.DEBUG)
```

或设置环境变量：

```bash
export LOGURU_LEVEL=DEBUG
```

## 当前阶段状态

### 已完成（Phase 3）

1. **LLM 诊断器**: `LLMDiagnoser` 将 traces 序列化后发送给 LLM，由 LLM 分析失败模式返回结构化诊断。LLM 调用失败时自动降级为规则模式。
2. **LLM 补丁规划器**: `LLMPatchPlanner` 由 LLM 根据诊断和 traces 决定具体的 override 字段和值，包括自由文本 `system_hint`。LLM 调用失败时自动降级为规则模式。
3. **PatchExecutor 兼容**: `_parse_intent` 支持 LLM 输出的 JSON 格式（直接使用 LLM 决定的字段值）和旧的关键词格式。
4. **真实金丝雀评估**: `CanaryRunner` 运行真正的 Harbor Trial 对比 baseline vs candidate reward。
5. **并发金丝雀**: baseline 和 candidate 两轮 trials 全部并发执行（`asyncio.gather`），8 个 trial 时间接近 1x。
6. **任务自动选择**: 从累积 traces 中自动提取失败任务，补充成功任务用于回归检测。
7. **灵活配置**: 设置 `llm_model` 即启用 LLM 模式，留空则使用规则模式。支持任意 OpenAI-compatible API。

### 已完成（Meta-RL 训练集成）

8. **Meta-RL 独立训练步**: LLM 的诊断/规划对话被自动录制，根据金丝雀评估结果赋予 reward（promoted → 正 reward，rejected → 负 reward），tokenize 后作为独立 mini batch 在主 RL step 之后执行一个额外的梯度更新。不与主 batch 混合，不受主 batch 大小限制，可独立扩展。
9. **复用本地推理引擎**: Meta LLM 默认复用 SkyRL 的 vLLM 推理引擎（同模型、同端口），无需额外部署。

### 当前限制

1. **补丁范围**: 仅通过 YAML overrides 修改行为。不修补 Python 源代码。
2. **并行化模块**: 尚不允许修补 `parallelization_policy` 和 `merge_policy`。
3. **单次诊断**: 每个周期独立诊断，不跨周期累积历史上下文。
4. **Meta-RL reward 粒度**: 同一周期内所有 LLM 对话共享相同 reward（基于整体金丝雀结果），未区分诊断和规划各自的贡献。

## Meta-RL 训练机制

### 原理

模型同时接收两类训练信号：

| 任务类型 | Prompt | Response | Reward 来源 |
|---------|--------|----------|------------|
| 终端任务 | 任务描述 | 多轮 agent 对话 | Verifier 评分 |
| 元推理任务 | Trace 分析 / 补丁规划 prompt | JSON 诊断 / override 值 | 金丝雀 delta_score |

### Reward 赋值规则

- 如果本周期有至少 1 个 candidate 被 promoted：`reward = max(delta_score, 0.1)`
- 如果所有 candidate 被 rejected：`reward = -0.5`
- 如果没有 candidate 生成：不产生训练样本

### 数据流

```
LLM 诊断 (system+user→assistant JSON)  ─┐
                                          ├─ 录制对话
LLM 规划 (system+user→assistant JSON)  ─┘
                ↓
        金丝雀评估 → delta_score
                ↓
        赋予 reward → tokenize → 存入 generator._pending_meta_batch
                ↓
        ┌─────────────────────────────────────────┐
        │ Trainer 主循环 Step N:                    │
        │  1. 主 RL batch (256 样本) → train_step  │
        │  2. Meta mini batch (N 样本) → train_step │ ← 独立梯度更新
        │  3. 同步权重 → 下一轮推理                  │
        └─────────────────────────────────────────┘
                ↓
        下一个 batch：模型在终端任务上更强，
        同时在元推理任务上也更强
```

### 为什么采用独立训练步

1. **不受主 batch 大小限制**: Meta 样本数量可任意扩展（4→16→64）而不影响主训练
2. **零侵入验证**: 不触发 SkyRL 的 `validate_generator_output` 的 prompts==responses 检查
3. **GRPO 独立分组**: Meta 样本内部形成独立 GRPO group，不干扰终端任务的 advantage 计算
4. **独立监控**: Wandb 中以 `meta_rl/*` 前缀单独记录 loss、batch_size 等指标

### 配置

默认配置已自动复用本地 vLLM 推理引擎：

```bash
# llm_model 默认取 SERVED_MODEL_NAME（本地模型）
# llm_base_url 默认取 http://127.0.0.1:8000/v1（推理引擎端口）
# 无需额外配置即可启用 meta-RL
```

如需使用外部强模型做元推理（不参与 RL 训练）：

```bash
META_LLM_MODEL=gpt-4o META_LLM_API_KEY=sk-xxx bash run_megatron.sh
```

## 未来增强（Phase 4+）

- **Python AST 修补**: 允许修补 Harbor Terminus2 Python 源代码
- **并行化策略**: 启用 `parallelization_policy` 和 `merge_policy` 的修补
- **多 Agent 协调**: 跨多个 agent 实例的元学习
- **自适应间隔**: 根据失败率动态调整 `interval_batches`
- **跨周期上下文**: 让 LLM 看到之前周期的诊断/补丁历史
- **细粒度 meta reward**: 分别评估诊断和规划的质量

## 故障排查

### 元学习周期未运行

**症状**: 日志中没有 `meta/cycle_batch` 指标

**原因**:
1. `generator.meta.enabled=false`（检查配置）
2. `interval_batches` 太大（例如 1000 但总共只有 100 个批次）
3. Agent 名称不以 "terminus" 开头（元学习仅对 Terminus agents 启用）

**修复**: 设置 `generator.meta.enabled=true` 并减少 `interval_batches`

### 未生成补丁

**症状**: 日志中 `meta/num_candidates=0`

**原因**:
1. 未检测到失败（所有任务成功）
2. 失败模式不匹配任何诊断规则
3. 诊断的模块不在白名单中

**修复**: 检查 trace 记录的 `failure_tags`。如需要添加自定义诊断规则。

### 补丁被拒绝

**症状**: `meta/rejected > 0` 但 `meta/promoted=0`

**原因**:
1. 金丝雀评估 `delta_score <= 0`（candidate 表现不优于 baseline）
2. 检测到回归（`delta < -0.05`）
3. 金丝雀评估出错（exception 被捕获后标记为 regression）
4. 无可用的 canary 任务（所有 traces 缺少 `metadata.prompt`）

**修复**: 检查 canary 日志（搜索 "Canary" 关键字），确认任务路径是否有效、baseline 和 candidate reward 是否合理。

### 金丝雀评估耗时过长

**症状**: 元学习周期需要很长时间才完成

**原因**: 金丝雀评估并发运行 `2 * canary_num_tasks` 个 Harbor Trials（baseline + candidate 各一轮并发），但受 GPU 推理吞吐量和 rate limiter 约束。

**修复**: 减小 `canary_num_tasks`（例如 2），或增加 `interval_batches` 以降低触发频率，或增大 `max_concurrency`。

### LLM 诊断/规划失败

**症状**: 日志中出现 "LLM diagnosis failed" 或 "LLM patch planning failed"

**原因**:
1. `llm_base_url` 不可达或模型服务未启动
2. `llm_model` 名称不匹配
3. API key 无效
4. LLM 返回了无法解析的 JSON

**修复**: 检查 LLM 服务是否可用（`curl {llm_base_url}/models`），确认模型名称正确。系统会自动降级为规则模式，不影响训练。

### Overrides 未应用

**症状**: 补丁提升后 agent 行为未改变

**原因**:
1. Override 文件未被 Terminus2 读取（检查 `meta_overrides` 注入）
2. Override 字段无效果（例如 `system_hint=""`）
3. Override 格式不匹配（嵌套 vs 扁平）

**修复**: 检查 Harbor 日志中的 "Loaded meta_overrides" 调试消息。验证任务指令中的 override 部分。

## Terminus-1 → Terminus-2 进化实验

### 核心思路

Terminus-1 是一个极简的终端 agent，而 Terminus-2 拥有丰富的机制（两步确认、上下文摘要、解析错误反馈、自我反思等）。通过 meta-learning，我们让模型**自主发现** Terminus-1 的不足，并通过 hook 代码注入来**自主实现**改进，逐步向 Terminus-2 的能力靠近。

这不是预设的"能力开关"，而是真正的代码进化——模型需要：
1. 通过诊断器分析轨迹，发现 Terminus-1 的具体问题（如过早完成任务、解析失败无恢复等）
2. 通过 planner 编写 Python hook 代码来解决这些问题
3. 通过金丝雀评估验证改进效果

### Terminus-1 与 Terminus-2 的关键差异

| 特性 | Terminus-1 | Terminus-2 |
|------|-----------|-----------|
| 完成确认 | `is_task_complete=true` 立即退出 | 两步确认（需要连续两次确认） |
| 解析错误 | 静默返回空命令 | 反馈给 LLM 重试 |
| 上下文管理 | 无 | 主动/被动摘要 |
| 自我反思 | 无 | 每 N 步注入反思 |
| 计划字段 | 无（只有 state_analysis） | analysis + plan |
| LLM 重试 | 无 | @retry(3) + 上下文溢出处理 |
| 输出截断 | 简单字节分割 | 保留重要行 |

### Hook 注入点

Terminus-1 支持与 Terminus-2 相同的 5 个 hook 注入点：

- `before_llm_call(prompt, context) -> str`：可注入反思提示、计划指令、上下文管理
- `before_execute(commands, context) -> list`：可过滤/重排命令、添加验证步骤
- `after_execute(terminal_output, context) -> str`：可智能过滤输出、高亮错误
- `on_timeout(command, output, context) -> str`：可改进超时恢复策略
- `on_parse_error(raw, error, context) -> str|None`：可构造重试提示

### Planner 的代码感知

当 agent 为 terminus-1 时，planner 的系统提示会自动切换为 Terminus-1 的实际源码上下文，包括：
- Terminus-1 的完整 agent loop 代码
- 明确列出 Terminus-1 缺少的功能
- 不预设解决方案，让模型根据诊断结果自主决定写什么 hook

### 运行方式

```bash
# 在 run_megatron.sh 中修改：
harbor_trial_config.agent.name=terminus-1
generator.meta.override_base=/home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus1
```

### 预期进化路径

模型可能自主发现并实现的改进（不保证顺序）：
- 在 `on_parse_error` 中实现错误反馈重试
- 在 `before_llm_call` 中注入反思和计划提示
- 在 `after_execute` 中实现智能输出截断
- 在 `before_execute` 中添加命令验证
- 在 `before_llm_call` 中实现简单的上下文管理

## 参考资料

- **设计文档**: `/home/ray/SkyRL/meta_learning_terminal_agent_design.md`
- **集成计划**: `/home/ray/SkyRL/terminus_integration_plan.md`
- **可修补模块规范**: `/home/ray/SkyRL/terminus_patchable_modules_spec.md`
- **Harbor 文档**: https://harborframework.com/docs
- **SkyRL 文档**: `/home/ray/SkyRL/README.md`

## 联系方式

如有问题或疑问，请提交 GitHub issue 或联系 SkyRL 团队。
