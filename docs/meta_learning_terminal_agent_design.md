# Terminal Agent Meta-Learning 设计方案

## 1. 目标

本文档提出一套用于 Terminal Agent 的 Meta-Learning 系统设计，目标是同时产出两类结果：

- 一个基于 SkyRL 和 Qwen3.5-4B 训练得到的 terminal-task 模型
- 一个可以持续演化的目标智能体，其能力升级路径类似于从 `terminus` 演化到 `terminus-2`

最终优化目标不是仅仅提升最近训练 batch 的 reward，而是让改进后的目标智能体在 Terminal Bench 2 上取得稳定、可验证的分数提升。

该设计重点强调：

- agent 改动的模块化
- agent 演化过程的可控与版本化
- 可复用的 toolkit 抽象，包括观测、诊断、编辑、验证四类能力
- 只手写可演化骨架，不手写具体策略逻辑

## 2. 目标对象与适用范围

### 2.1 最终要改造的对象

本文档里的最终目标对象应当是 `Terminus` 或 `Terminus-2` 这类目标智能体，而不是把 `SkyRL` 自己改造成最终交付智能体。

更准确地说：

- `SkyRL` 负责训练、采样、评测、版本管理与 meta-learning orchestration
- `Terminus` 或 `Terminus-2` 负责作为被改造、被评测、被版本化升级的目标智能体
- `Meta-Learning Toolkit` 负责在两者之间建立诊断、patch、A/B 评测、promote 的闭环

因此，最终交付结果应当包括：

- 一个训练后的模型
- 一个经过模块化改造和版本升级后的 `Terminus` 智能体

### 2.2 当前仓库中的关系

当前仓库里最直接可接入的边界仍然在 SkyRL 侧，例如：

- Harbor 训练入口：
  - [/home/ray/SkyRL/examples/train_integrations/harbor/entrypoints/main_harbor.py](/home/ray/SkyRL/examples/train_integrations/harbor/entrypoints/main_harbor.py)
- SkyRL agent 集成入口：
  - [/home/ray/SkyRL/skyrl-agent/skyrl_agent/integrations/skyrl_train/skyrl_train_main.py](/home/ray/SkyRL/skyrl-agent/skyrl_agent/integrations/skyrl_train/skyrl_train_main.py)
- Agent runtime 加载器：
  - [/home/ray/SkyRL/skyrl-agent/skyrl_agent/auto.py](/home/ray/SkyRL/skyrl-agent/skyrl_agent/auto.py)
- 基础 AgentRunner：
  - [/home/ray/SkyRL/skyrl-agent/skyrl_agent/agents/base.py](/home/ray/SkyRL/skyrl-agent/skyrl_agent/agents/base.py)

这意味着第一阶段可以先借助 SkyRL 提供训练与 orchestration 基座，但 patch 的目标应逐步转向 `Terminus` 的模块，而不是长期把演化对象绑死在 `skyrl-agent` 上。

### 2.3 当前训练入口

本文档假设当前训练入口为：

- [/home/ray/SkyRL/examples/train_integrations/harbor/run_megatron.sh](/home/ray/SkyRL/examples/train_integrations/harbor/run_megatron.sh)

本方案不要求一开始就重写 PPO 内核，而是在 generator 与目标 agent runtime 的边界之外增加一层 versioned meta-controller。

## 3. 核心设计原则

### 3.1 双层循环

系统分为两层循环：

- 内环：RL 训练循环，负责 policy 优化
- 外环：meta-learning 循环，负责目标 agent 的结构与策略升级

### 3.2 自演化逻辑原则

本方案的核心原则是：

- 人工手写“可演化骨架”
- 不人工手写“具体策略逻辑”

也就是说，工程上应预先定义：

- 可 patch 的模块边界
- 这些模块的接口契约
- patch 白名单和安全限制
- 验证与 promote 规则
- trace 与评测信号

但不预先写死：

- 什么时候应该并发
- 什么时候应该重试
- 什么时候应该 finish
- 怎样做 verification
- 怎样压缩上下文
- 怎样做 memory 管理

这些具体行为应当由模型通过 patch 在允许的模块内自行演化出来。

### 3.3 对异步逻辑的要求

异步逻辑也遵守同样原则。

也就是说，我们不应手写“具体的异步策略逻辑”，而应只手写：

- 异步能力的接口层
- 子任务的数据结构
- 并发执行的安全约束
- 聚合与冲突控制机制

然后让模型自己决定：

- 是否需要 fanout
- 哪些步骤适合并发
- 应该拆出哪些 subtask
- 如何合并多个分支的结果

因此，异步应被视为一种“可演化能力”，而不是预写死的固定策略。

## 4. 总体架构

### 4.1 内环

内环是现有的 RL 训练循环。

- 使用 SkyRL 在 terminal 轨迹上训练 Qwen3.5-4B
- 由 Harbor 或其它 task backend 负责采样轨迹
- 收集 reward、日志、工具使用记录、失败样例等信息
- policy checkpoint 按现有方式持续更新

### 4.2 外环

外环是面向目标智能体改进的 meta-learning 循环。

- 每经过 `K` 个 batch 或每 `N` 个 global step，触发一次 meta-cycle
- 从近期 rollout 中抽取失败样例与近失败样例
- 对失败模式进行聚类和诊断
- 在 patchable target-agent modules 上规划候选改动
- 生成候选代码 diff 或配置 diff
- 运行 smoke test 和 A/B 评测
- 决定是否接受某个 patch
- 如果接受，则提升出新的 `agent_version`
- 之后继续使用新版本 agent 进行采样和训练

整个系统形成两种时间尺度：

- 快时间尺度：policy 优化
- 慢时间尺度：agent 架构与行为策略升级

## 5. 目标 Pipeline

### 5.1 Meta-Cycle Pipeline

1. 使用 SkyRL 采样一个训练 batch。
2. 收集 trajectories、rewards、terminal outcomes、tool logs 和相关 metadata。
3. 将 batch 转换为结构化诊断工件。
4. 对失败样例进行聚类，识别重复出现的 failure mode。
5. 选择一个 failure cluster，并生成 root cause hypotheses。
6. 将问题路由到 patchable target-agent modules。
7. 生成一个或多个 patch plan。
8. 生成一个或多个结构化 patch candidate。
9. 运行本地校验：
   - import 和 compile 检查
   - smoke task 检查
   - patch schema 检查
10. 在固定任务子集上运行 canary A/B evaluation。
11. 对比 before / after 指标。
12. 做出决策：
   - accept
   - reject
   - retry with narrower scope
13. 如果接受：
   - 注册新的 `agent_version`
   - 更新当前 active serving agent
   - 后续 rollout 与训练切换到新版本
14. 记录所有中间产物，供后续 meta-training 使用。

### 5.2 Promotion 规则

一个 patch 只有在满足以下条件时才允许提升：

- 通过 patch safety checks
- 未修改 forbidden modules
- 通过 smoke validation
- canary evaluation 的主指标提升超过阈值
- 在 stable subset 上没有超过容忍度的回退
- reviewer 给出的风险分数在可接受预算内

## 6. 系统核心实体

### 6.1 AgentVersion

表示一个具体的目标 agent runtime 版本。

字段建议：

- `agent_version_id`
- `parent_agent_version_id`
- `base_commit`
- `patch_id`
- `code_hash`
- `config_hash`
- `created_at`
- `promotion_reason`
- `metrics_snapshot`

### 6.2 PolicyVersion

表示 rollout 或训练时使用的模型 checkpoint。

字段建议：

- `policy_version_id`
- `global_step`
- `checkpoint_path`
- `served_model_name`

### 6.3 MetaCycle

表示一次完整的 diagnose-to-promote 事务。

字段建议：

- `meta_cycle_id`
- `trigger_step`
- `input_agent_version`
- `input_policy_version`
- `selected_failure_cluster`
- `candidate_patch_ids`
- `decision`
- `output_agent_version`

### 6.4 PatchableModule

表示一个允许被 meta-system 修改的目标 agent 模块。

字段建议：

- `name`
- `owned_files`
- `interface_contract`
- `allowed_change_types`
- `required_tests`
- `forbidden_dependencies`
- `risk_level`

## 7. Failure Taxonomy

为了让诊断结果可训练、可比较、可统计，所有失败模式都应映射到一个受控 taxonomy 上。

推荐初始 taxonomy：

- `planning_failure`
- `tool_selection_failure`
- `tool_execution_failure`
- `verification_failure`
- `recovery_failure`
- `termination_failure`
- `memory_failure`
- `context_overload`
- `async_coordination_failure`
- `sync_bottleneck`
- `reward_misalignment`

其中 `sync_bottleneck` 用于显式标记“同步执行过慢，本可以并发”的情况，为后续模型自主演化异步逻辑提供监督信号。

## 8. Toolkit 设计

toolkit 是整个项目中最值得沉淀的可复用资产。它不应只是一个大 prompt，而应设计为可扩展的工程组件集合。

建议目录：

```text
meta_toolkit/
  observability/
    trace_schema.py
    trace_store.py
    summarizer.py
    timeline.py
    diffing.py
    clustering.py
  diagnosis/
    taxonomy.py
    root_cause.py
    attribution.py
    patchability.py
  editing/
    module_registry.py
    patch_schema.py
    planner.py
    patcher.py
    sandbox_workspace.py
    whitelist.py
  validation/
    smoke.py
    canary.py
    benchmark.py
    comparator.py
    promotion.py
  runtime/
    meta_controller.py
    version_registry.py
    event_bus.py
    capability_runtime.py
  data/
    meta_dataset.py
    patch_outcomes.py
    preference_pairs.py
```

### 8.1 Observability Toolkit

目标：

- 让模型和系统都能高效“看到”轨迹中的问题
- 将原始轨迹转换为结构化、可诊断的工件
- 显式暴露同步瓶颈、等待时间和可并发机会

推荐工具：

- `trace.search`
- `trace.get`
- `trace.summary`
- `trace.diff`
- `trace.cluster_failures`
- `trace.timeline`
- `trace.stats`
- `trace.slice`
- `trace.find_decision_junctions`

核心 trace schema 应额外记录：

- step latency
- idle wait time
- tool latency
- read-only opportunity
- fanout opportunity
- timeout source

### 8.2 Diagnosis Toolkit

目标：

- 将轨迹表征出的症状映射为 root-cause hypotheses
- 判断应该修改目标 agent 的哪个模块
- 降低由于诊断不充分而导致的低质量 patch

推荐工具：

- `diagnose.root_cause`
- `diagnose.module_attribution`
- `diagnose.counterexample`
- `diagnose.failure_taxonomy`
- `diagnose.patchability`

建议输出格式：

```json
{
  "problem_type": "sync_bottleneck",
  "root_cause_hypotheses": [
    "agent performs repo scan and log inspection serially",
    "verification path is fully sequential although subtasks are read-only"
  ],
  "candidate_modules": ["parallelization_policy", "merge_policy"],
  "confidence": 0.81
}
```

### 8.3 Agent Editing Toolkit

目标：

- 让目标 agent 的改动安全、模块化、可审计
- 将修改范围限制在批准的模块中
- 让 patch 成为可复用的训练样本

推荐工具：

- `agent.inspect_module`
- `agent.list_patchable_modules`
- `agent.read_file`
- `agent.propose_patch`
- `agent.apply_patch_candidate`
- `agent.validate_patch_schema`
- `agent.explain_patch`

#### 8.3.1 Patch Scope

第一阶段只允许 patch 以下模块：

- planner policy
- retry policy
- finish policy
- verification policy
- context summarizer
- memory policy
- parallelization policy
- merge policy
- 指定 prompt 或 YAML 模块

第一阶段禁止 patch 以下模块：

- evaluator
- reward function
- benchmark harness
- metric comparator
- core promotion gate
- training core
- dataset contents
- 任意未注册共享状态写入路径

#### 8.3.2 Patch Schema

patch 应先表示为结构化 schema，再转化为代码 diff。

示例：

```json
{
  "target_modules": ["parallelization_policy", "merge_policy"],
  "files": [
    {
      "path": "terminus/policies/parallelization_policy.py",
      "change_type": "modify",
      "intent": "fan out read-only repo scan and log inspection before planning"
    }
  ],
  "risk": "medium",
  "requires_tests": ["smoke_terminal", "tb2_canary"],
  "rollback_if": [
    "shared_state_conflict_rate > 0",
    "main_score_drop > 1%"
  ]
}
```

### 8.4 Validation and Promotion Toolkit

目标：

- 判断 agent 修改是否真的有价值
- 防止 reward hacking 和局部过拟合
- 决定是否提升出新版本 agent

推荐工具：

- `eval.smoke`
- `eval.canary`
- `eval.benchmark`
- `eval.compare`
- `eval.regression_check`
- `promote.decide`

建议追踪的指标包括：

- main terminal benchmark score
- solve rate
- average reward
- average turn count
- tool error rate
- retry recovery rate
- timeout rate
- verification-before-finish ratio
- wall-clock latency
- idle wait time
- async branch success rate
- shared-state conflict rate

## 9. 自演化模块化设计

为了做到“由模型新增逻辑，而不是人工写死逻辑”，目标 agent 应拆成可演化策略模块。

推荐模块：

- `planner_policy`
- `tool_policy`
- `retry_policy`
- `verification_policy`
- `finish_policy`
- `memory_policy`
- `context_summarizer`
- `parallelization_policy`
- `merge_policy`
- `state_manager`
- `conflict_guard`

每个模块应暴露：

- purpose
- owned files
- configuration surface
- runtime interface
- evaluation hooks
- patch safety limits

接口示意：

```python
class PatchableModule:
    name: str
    owned_files: list[str]
    interface_contract: str
    allowed_change_types: list[str]
    required_tests: list[str]
    forbidden_dependencies: list[str]
```

对于异步相关逻辑，建议只手写接口骨架，不手写具体策略实现，例如：

```python
class ParallelizationPolicy:
    def should_fanout(self, state, trace_summary) -> bool: ...
    def propose_subtasks(self, state, trace_summary) -> list[dict]: ...
    def choose_merge_policy(self, state, branch_results) -> str: ...
```

这样模型 patch 的是策略实现，而不是任意改动整个执行引擎。

## 10. 多 Agent Meta Roles

AOrchestra 风格的多角色设计非常适合放进 meta-cycle，但每个角色的职责必须尽量收敛、结构化。

### 10.1 Diagnoser

输入：

- batch trajectory summaries
- reward statistics
- failure examples
- 固定 failure taxonomy

输出：

- root cause hypotheses
- problem type
- candidate modules
- confidence

### 10.2 PatchPlanner

输入：

- diagnoser JSON
- module registry
- file whitelist
- module responsibilities

输出：

- patch plan
- files to touch
- risk level
- required tests

### 10.3 Patcher

输入：

- patch plan
- exact file contents
- patch constraints

输出：

- structured patch
- candidate diff
- self-check notes

### 10.4 PatchTester / Reviewer

输入：

- patch diff
- compile 与 import 结果
- smoke test 结果
- canary before / after 指标

输出：

- ship score
- risk tags
- missing coverage
- recommendation

### 10.5 Promoter

输入：

- A/B comparison
- reviewer score
- regression analysis

输出：

- accept 或 reject
- promotion scope
- rollback conditions

## 11. 集成方案

### 11.1 推荐集成策略

不要一开始就直接改 PPO 内核，而是在 generator 与目标 agent runtime 边界旁边增加一个 sidecar meta-controller。

建议集成点：

- 在 generator 或 agent integration 层增加 `MetaController`
- 给 rollout output 增加 metadata：
  - `agent_version`
  - `policy_version`
  - `meta_cycle_id`
  - `failure_tags`
  - `sync_bottleneck_score`
- 将 trajectory artifacts 写入 trace store
- 通过训练进度回调或单独 coordinator 进程周期性触发 outer loop

### 11.2 Terminus 的接入方式

如果最终要产出的是改造后的 `Terminus` 或 `Terminus-2`，则工程上应优先把 `Terminus` 的策略逻辑重构为 patchable modules，再让 SkyRL 的 meta-learning loop 去 patch 它。

也就是说：

- `SkyRL` 是训练与 orchestration 基座
- `Terminus` 是被修改、被评测、被提升版本的对象
- `meta_toolkit` 是连接两者的中间层

如果第一阶段暂时只能在 `skyrl-agent` 上验证流程，也应把它视为原型载体，而不是最终长期目标对象。

## 12. Patch 接受逻辑

patch 的 accept 依据必须是 holdout improvement，而不是最近 batch reward。

建议规则：

```text
accept if
  delta_main_score > tau_main
  and delta_cluster_score > tau_cluster
  and regression_rate_on_stable_subset < tau_regress
  and reviewer_risk <= risk_budget
```

对于异步相关 patch，还应额外满足：

- `wall_clock_latency` 不恶化
- `shared_state_conflict_rate == 0`
- `branch_merge_failure_rate` 在阈值内

## 13. 分阶段实施计划

### Phase 1: Instrumentation

交付物：

- trajectory trace schema
- trace store
- batch summaries
- failure taxonomy
- sync bottleneck signals

### Phase 2: Modular Agent Surface

交付物：

- patchable module registry
- file whitelist
- patch schema validator
- version registry
- Terminus policy interface layer

### Phase 3: Evaluation Gates

交付物：

- smoke validation
- canary evaluation
- comparator
- promotion gate

### Phase 4: Meta-Cycle Execution

交付物：

- diagnoser
- patch planner
- patcher
- reviewer
- promoter

### Phase 5: Self-Evolving Logic

交付物：

- self-evolving retry logic
- self-evolving verification logic
- self-evolving finish logic
- self-evolving memory logic
- self-evolving parallelization logic

## 14. MVP 建议

第一版工作版本应尽量收敛。

MVP 范围建议：

- 先把目标 agent 明确为 `Terminus` 或 `Terminus-2`
- 仅允许 patch planner、retry、verification、finish、parallelization 模块
- 每固定数量 batch 触发一次 meta-cycle
- 从近期失败样本中选取 top failed examples
- 每次最多生成 3 个 patch candidate
- 只运行 smoke 与 canary validation
- 只有在提升明显时才 promote
- 保存所有中间工件，供后续训练使用

## 15. 风险与缓解策略

### 15.1 Reward Hacking

风险：

- 某个 patch 提高了局部 reward，但并没有提升 Terminal Bench 2 分数

缓解：

- 要求通过 canary 或 holdout benchmark improvement
- 禁止修改 evaluator 与 reward modules

### 15.2 近期失败样例过拟合

风险：

- patch 修好了一个 cluster，却损害了整体泛化表现

缓解：

- 使用 stable regression sets
- 跟踪 per-cluster metrics

### 15.3 代码漂移失控

风险：

- meta-system 持续进行无约束修改，导致 agent 行为越来越不可控

缓解：

- 使用 patchable module registry
- 强制 patch schema 校验
- 保留 promotion gate 与 rollback 路径

### 15.4 异步演化失控

风险：

- 模型在没有安全壳的情况下直接向任意代码注入并发逻辑

缓解：

- 不允许自由修改底层 executor
- 只允许在注册的 `parallelization_policy` 与 `merge_policy` 中演化异步逻辑
- 强制共享状态写入经过 `state_manager` 与 `conflict_guard`

## 16. 最终建议

最值得投入的方向，不是把 meta-learning 当作一次性训练技巧，而是把它做成一套基础设施。

项目真正应该沉淀的核心资产，是一个可复用的 `Meta-Learning Agent Toolkit`，其核心能力包括：

- 结构化轨迹观测
- 面向根因的失败诊断
- 模块化且安全的 agent 编辑
- 基于证据的验证与 promotion
- 支持模型自行演化具体逻辑的接口骨架

在这套 toolkit 之上，项目可以同时产出两个关键结果：

- 一个同时学习 terminal-task 行为与 meta-improvement 行为的模型
- 一个经过模块化改造、可持续升级、并能从 `Terminus` 演化到更强版本的目标智能体

## 17. 下一步建议

建议的近期实施顺序：

1. 明确目标智能体仓库与模块边界，优先面向 `Terminus` 设计 patch surface
2. 实现 trace schema 与 trace store，并加入 sync bottleneck signals
3. 定义 failure taxonomy 与 patchable module registry
4. 抽出 Terminus 的 policy interface layer
5. 实现最小版 promotion gate，包括 smoke 和 canary checks
6. 在 rollout metadata 中加入 `agent_version` 与 `policy_version`
7. 先在 planner、retry、verification、parallelization 模块上原型化自演化 patch 闭环
