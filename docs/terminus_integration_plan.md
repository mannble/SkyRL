# Terminus 集成实施方案

## 1. 文档目标

本文档用于回答一个更具体的工程问题：

如何将 `Terminus` 或 `Terminus-2` 改造成一个可以被 SkyRL + Meta-Learning 系统持续观测、诊断、修改、评测和升级的目标智能体。

该文档是对 [meta_learning_terminal_agent_design.md](/home/ray/SkyRL/docs/meta_learning_terminal_agent_design.md) 的落地补充，重点不再讨论总体理论设计，而是聚焦于：

- `Terminus` 应如何模块化
- 哪些部分需要先抽象成可 patch 接口
- SkyRL 应如何接入 `Terminus`
- 第一阶段如何最小改造并跑通闭环
- 后续如何逐步放开更强的自演化能力

## 2. Terminus 在系统中的定位

在整个系统中，`Terminus` 应被视为：

- 被训练模型调用的目标 agent runtime
- 被 trace、diagnose、patch、evaluate 的主要对象
- 被版本化管理的智能体
- 最终交付的智能体产物之一

而 `SkyRL` 的角色应被限制为：

- 训练和采样基座
- rollout 和 reward 收集系统
- meta-controller 的承载框架
- patch 验证和 promotion 的控制中枢

因此，系统关系可以表达为：

```text
Qwen3.5-4B policy/model
        |
        v
SkyRL training + rollout + meta-controller
        |
        v
Terminus runtime (patch target)
        |
        v
Terminal Bench 2 / task environments / evaluators
```

## 3. 集成目标

Terminus 的集成不只是“能被调用”，而是要满足以下工程目标：

1. 能被 SkyRL 作为 generator runtime 或 agent runtime 调用
2. 能输出足够完整的 trajectory / tool / timing / state 信息
3. 能被拆成 patchable modules
4. 能做版本化升级与回滚
5. 能接受模型生成的 patch，并进行模块级验证
6. 最终可以作为独立智能体复用，而不是永久依赖 SkyRL 内部实现

## 4. Terminus 需要具备的能力表面

为了让 meta-learning 真正能改它，Terminus 必须提供足够明确的“能力表面”。

建议至少包含以下五类表面：

### 4.1 Runtime Surface

让 SkyRL 可以运行它。

需要包含：

- 标准化的 agent 启动入口
- 单任务执行入口
- 多回合交互入口
- tool call 和 observation 的统一事件流
- 中间状态导出能力

### 4.2 Trace Surface

让 meta-system 可以看懂它。

需要包含：

- 每一步 assistant action
- 每一次 tool 调用参数
- 每次 tool observation
- 每步 latency
- 每步 finish/retry/verify 决策来源
- 当前上下文摘要状态
- 错误与 timeout 原因

### 4.3 Patch Surface

让模型能够改它。

需要包含：

- patchable module registry
- 每个模块的 owned files
- 每个模块的 interface contract
- patch 白名单
- patch schema validator

### 4.4 Eval Surface

让系统可以稳定地评测它。

需要包含：

- smoke task 套件
- canary benchmark 子集
- before/after A/B 对比能力
- 模块级 regression check

### 4.5 Packaging Surface

让改造后的 Terminus 可以独立复用。

需要包含：

- agent version manifest
- config export
- model compatibility metadata
- 可独立启动的推理入口

## 5. 推荐的模块化拆分

如果 Terminus 当前还是一个比较集中的 agent 实现，建议优先拆成以下模块。

### 5.1 第一层：核心策略模块

这是最适合优先开放给 meta-learning patch 的部分。

- `planner_policy`
- `tool_policy`
- `retry_policy`
- `verification_policy`
- `finish_policy`
- `memory_policy`
- `context_summarizer`
- `parallelization_policy`
- `merge_policy`

这些模块主要负责“策略”，适合让模型自己演化。

### 5.2 第二层：受控运行模块

这些模块通常不建议一开始开放自由 patch，但需要先抽象清楚接口。

- `state_manager`
- `conflict_guard`
- `tool_router`
- `execution_coordinator`
- `observation_normalizer`

这些模块主要负责系统安全性和运行稳定性。

### 5.3 第三层：冻结基础设施模块

这些模块第一阶段应视为冻结区，不应允许模型直接改动。

- 底层 shell executor
- benchmark harness
- reward function
- evaluator
- promotion gate
- trace store
- metric aggregator

## 6. 推荐的接口设计

为了做到“只手写骨架，不手写策略逻辑”，建议 Terminus 暴露一组统一的 policy interface。

### 6.1 基础 PatchableModule 接口

```python
class PatchableModule:
    name: str
    owned_files: list[str]
    interface_contract: str
    allowed_change_types: list[str]
    required_tests: list[str]
    forbidden_dependencies: list[str]
```

### 6.2 PlannerPolicy

```python
class PlannerPolicy:
    def build_plan(self, state, task, context) -> dict: ...
```

### 6.3 RetryPolicy

```python
class RetryPolicy:
    def should_retry(self, state, last_step, error_info) -> bool: ...
    def next_retry_action(self, state, error_info) -> dict: ...
```

### 6.4 VerificationPolicy

```python
class VerificationPolicy:
    def should_verify(self, state, candidate_result) -> bool: ...
    def build_verification_step(self, state, candidate_result) -> dict: ...
```

### 6.5 FinishPolicy

```python
class FinishPolicy:
    def should_finish(self, state) -> bool: ...
    def finish_reason(self, state) -> str: ...
```

### 6.6 ParallelizationPolicy

```python
class ParallelizationPolicy:
    def should_fanout(self, state, trace_summary) -> bool: ...
    def propose_subtasks(self, state, trace_summary) -> list[dict]: ...
    def choose_merge_policy(self, state, branch_results) -> str: ...
```

### 6.7 MergePolicy

```python
class MergePolicy:
    def merge(self, state, branch_results) -> dict: ...
```

这些接口的意义在于：

- 人工负责定义模块边界
- 模型负责修改实现逻辑
- runtime 负责调用这些逻辑

## 7. SkyRL 与 Terminus 的对接方式

推荐采用“薄接入层”方式，而不是把 Terminus 深度嵌入 SkyRL 内核。

### 7.1 推荐结构

```text
SkyRL
  -> MetaController
  -> TerminusAdapter
  -> Terminus Runtime
```

其中：

- `MetaController` 负责诊断、patch、评测、promote
- `TerminusAdapter` 负责把 SkyRL 的 generator input/output 与 Terminus 的运行接口适配起来
- `Terminus Runtime` 负责真正执行 terminal agent 任务

### 7.2 TerminusAdapter 的职责

建议单独做一个适配层，职责包括：

- 把 SkyRL 的 prompt / env / sampling metadata 转成 Terminus 的 task input
- 运行 Terminus
- 收集 trajectory、tool events、timing metrics、finish reason
- 输出符合 SkyRL generator output 预期的数据
- 追加 `agent_version`、`policy_version`、`trace_id` 等 metadata

### 7.3 为什么要单独做 Adapter

因为这样有三个好处：

- 不污染 SkyRL 训练核心
- 不要求 Terminus 完全服从 SkyRL 的内部数据结构
- 后面 Terminus 可以独立复用，SkyRL 也能继续替换别的 agent

## 8. Trace 集成建议

为了让 meta-learning 能看懂 Terminus 的问题，必须先把 trace 打通。

### 8.1 每条 trajectory 至少记录

- `task_id`
- `agent_version`
- `policy_version`
- `turn_count`
- `tool_calls`
- `tool_failures`
- `step_latency`
- `idle_wait_time`
- `finish_reason`
- `verification_attempted`
- `retry_attempted`
- `sync_bottleneck_score`
- `failure_tags`

### 8.2 对异步演化特别重要的字段

如果以后希望模型自己长出异步逻辑，则应额外记录：

- 哪一步是 read-only
- 哪一步存在可并发机会
- 串行等待时间
- 是否因为同步执行过慢导致 timeout
- fanout 后的 branch 成功率
- merge 是否失败

## 9. Patch Surface 设计

为了让 Terminus 成为真正的 patch target，需要先明确 patch surface。

### 9.1 Patch 白名单建议

第一阶段开放：

- `terminus/policies/planner_policy.py`
- `terminus/policies/retry_policy.py`
- `terminus/policies/verification_policy.py`
- `terminus/policies/finish_policy.py`
- `terminus/policies/parallelization_policy.py`
- `terminus/policies/merge_policy.py`
- `terminus/prompts/`
- `terminus/configs/`

### 9.2 Patch 禁区建议

第一阶段禁止：

- `terminus/executors/`
- `terminus/evaluators/`
- `terminus/benchmarks/`
- `terminus/reward/`
- `terminus/core/runtime_loop.py` 中的关键安全逻辑

### 9.3 Patch 校验要求

每个 patch 必须满足：

- 仅触碰白名单文件
- 不引入未批准依赖
- 通过 import / compile checks
- 通过 smoke tests
- 通过 canary A/B evaluation

## 10. Versioning 设计

为了让 Terminus 真正成为可升级对象，版本系统必须独立存在。

### 10.1 Agent Version Manifest

每个版本建议导出 manifest，例如：

```json
{
  "agent_version": "terminus_v12",
  "parent_version": "terminus_v11",
  "patch_id": "patch_20260410_01",
  "code_hash": "...",
  "config_hash": "...",
  "compatible_model": "Qwen3.5-4B",
  "metrics": {
    "tb2_score": 0.34,
    "solve_rate": 0.27
  }
}
```

### 10.2 版本切换原则

- rollout 时始终显式记录 agent version
- patch 被 accept 后才提升默认 serving version
- 保留快速 rollback 到上一个稳定版本的能力

## 11. 第一阶段最小实施方案

为了尽快跑通闭环，建议先做最小可行改造。

### 11.1 MVP 目标

第一阶段只做这些事情：

- 让 SkyRL 能调用 Terminus
- 让 Terminus 输出结构化 trace
- 只开放少数策略模块给 patch
- 能在 fixed canary set 上做 before/after 比较
- 能 accept / reject 一个 patch 并产出新版本 manifest

### 11.2 MVP 只开放这些模块

- `planner_policy`
- `retry_policy`
- `verification_policy`
- `finish_policy`

如果你希望稍微激进一点，再加：

- `parallelization_policy`

### 11.3 MVP 暂时不做的事情

- 不开放底层 executor
- 不开放 reward/evaluator
- 不让模型自由新建复杂运行框架
- 不在第一阶段做大规模异步子图改写

## 12. 第二阶段扩展方向

第一阶段跑通后，再逐步开放更强能力。

### 12.1 扩展到自演化异步

前提：

- trace 中已经包含 sync bottleneck signals
- `parallelization_policy` 与 `merge_policy` 已抽象完成
- `state_manager` 与 `conflict_guard` 已稳定

届时允许模型：

- 新增只读 fanout 逻辑
- 调整 subtask 拆分规则
- 调整 merge 策略

### 12.2 扩展到 memory 和 context

可以逐步开放：

- `memory_policy`
- `context_summarizer`

### 12.3 扩展到更复杂的 planner

可以逐步允许 planner 生成多阶段计划，甚至多候选计划比较。

## 13. 实施顺序建议

推荐的工程顺序如下：

1. 确认 Terminus 仓库位置与当前主要 agent 入口
2. 为 Terminus 增加 `TerminusAdapter`
3. 打通 trajectory / timing / tool trace
4. 抽出 policy interface layer
5. 定义 patchable module registry
6. 加入 agent version manifest
7. 建立 smoke + canary A/B evaluation
8. 接入 SkyRL MetaController
9. 先在 planner / retry / verification 上跑 patch 闭环
10. 闭环稳定后再扩展到 parallelization

## 14. 成功标准

当以下条件满足时，可以认为 Terminus 集成基本成功：

- SkyRL 能稳定调用 Terminus 完成 rollout
- Terminus 的关键轨迹、工具、时延信息可以被 trace store 记录
- 至少 3 个策略模块可以被模型 patch
- patch 可以完成 smoke + canary 验证
- 系统可以 accept / reject patch 并产出新的 Terminus version
- 改造后的 Terminus 可以脱离 SkyRL 独立运行和复用

## 15. 最终建议

如果最终目标是产出“改造后的 Terminus 智能体”，那么工程重点就不应该放在长期演化 `skyrl-agent`，而应该放在：

- 把 SkyRL 作为训练与 meta-learning 的基座
- 把 Terminus 作为真正的 patch target 与交付对象
- 把 meta_toolkit 设计成既服务训练，也服务 Terminus 演化的中间层

这样最终产物才会清晰：

- 一个训练后的模型
- 一个模块化、可版本化、可复用、可持续升级的 Terminus 智能体
