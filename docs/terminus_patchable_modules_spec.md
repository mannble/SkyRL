# Terminus Patchable Modules 规格说明

## 1. 文档目标

本文档用于定义 `Terminus` 或 `Terminus-2` 在 Meta-Learning 系统中应暴露的可修改模块规格。

它回答的问题是：

- 哪些模块允许被模型 patch
- 每个模块负责什么
- 每个模块允许改哪些逻辑，不允许改哪些逻辑
- 每个模块应暴露怎样的接口
- 每个模块需要满足哪些测试和验证要求

本文档应与以下文档配套使用：

- [meta_learning_terminal_agent_design.md](/home/ray/SkyRL/docs/meta_learning_terminal_agent_design.md)
- [terminus_integration_plan.md](/home/ray/SkyRL/docs/terminus_integration_plan.md)

## 2. 总体原则

### 2.1 模块化原则

所有允许被模型修改的逻辑，都应落在显式注册的 patchable modules 中。

模型不应直接修改：

- 未注册模块
- 底层执行器
- evaluator
- reward 逻辑
- benchmark harness
- promotion gate
- trace store

### 2.2 自演化原则

人工手写的是：

- 模块边界
- 模块接口
- 文件归属
- 校验规则
- 测试要求

模型自行演化的是：

- 具体策略逻辑
- 具体 prompt 组织方式
- 具体 retry、verification、finish、parallelization 行为

### 2.3 最小开放原则

第一阶段只开放少量高价值策略模块。

建议第一阶段开放：

- `planner_policy`
- `retry_policy`
- `verification_policy`
- `finish_policy`
- `parallelization_policy`
- `merge_policy`

## 3. PatchableModule Registry 规范

每个 patchable module 都应在 registry 中注册。

建议数据结构：

```python
class PatchableModule:
    name: str
    description: str
    owned_files: list[str]
    interface_contract: str
    allowed_change_types: list[str]
    forbidden_change_types: list[str]
    required_tests: list[str]
    dependent_modules: list[str]
    risk_level: str
```

建议注册表输出中至少包含：

- 模块名
- 模块职责
- 文件归属
- 风险等级
- 所需 smoke tests
- 所需 canary tests
- 是否允许新增 helper 函数
- 是否允许新增配置项

## 4. 模块一：Planner Policy

### 4.1 职责

`planner_policy` 负责生成任务执行计划，决定高层子目标与动作顺序。

它关注的问题包括：

- 是否先看 repo 结构
- 是否先运行测试
- 是否先定位错误日志
- 是否要先做信息收集再执行修改

### 4.2 允许修改的内容

- 计划步骤生成逻辑
- 子目标拆分逻辑
- 不同任务类型下的 planning 模板
- plan 的提示词或结构化输出格式
- plan 的置信度或阶段切换条件

### 4.3 禁止修改的内容

- 底层工具执行器
- benchmark 接口
- 最终 reward 计算
- 共享状态底层读写路径

### 4.4 推荐接口

```python
class PlannerPolicy:
    def build_plan(self, state, task, context) -> dict: ...
    def revise_plan(self, state, feedback) -> dict: ...
```

### 4.5 建议文件归属

建议文件：

- `terminus/policies/planner_policy.py`
- `terminus/prompts/planner/`
- `terminus/configs/planner/`

### 4.6 验证要求

必须通过：

- `smoke_terminal`
- `plan_generation_smoke`
- `tb2_canary`

建议额外记录指标：

- planning turns
- 首次错误定位时间
- 计划后首次有效动作的延迟

## 5. 模块二：Retry Policy

### 5.1 职责

`retry_policy` 负责在失败、报错、测试未通过或工具异常后，判断是否继续尝试，以及如何尝试。

### 5.2 允许修改的内容

- retry 触发条件
- retry 次数上限
- 错误分类后的应对策略
- retry 时的下一步动作建议
- retry 提示词和策略模板

### 5.3 禁止修改的内容

- tool 的底层超时实现
- tool executor 的异常定义
- 全局最大安全预算

### 5.4 推荐接口

```python
class RetryPolicy:
    def should_retry(self, state, last_step, error_info) -> bool: ...
    def next_retry_action(self, state, error_info) -> dict: ...
```

### 5.5 建议文件归属

建议文件：

- `terminus/policies/retry_policy.py`
- `terminus/prompts/retry/`
- `terminus/configs/retry/`

### 5.6 验证要求

必须通过：

- `smoke_terminal`
- `retry_recovery_smoke`
- `tb2_canary`

建议额外记录指标：

- retry recovery rate
- 平均 retry 次数
- 无效 retry 比例

## 6. 模块三：Verification Policy

### 6.1 职责

`verification_policy` 负责判断在给出结论或 finish 前，是否需要做验证，以及如何验证。

### 6.2 允许修改的内容

- verification 触发条件
- verification 步骤生成策略
- 测试优先级选择
- finish 前强制验证规则
- verification prompt 和模板

### 6.3 禁止修改的内容

- 测试框架底层执行器
- evaluator 评分规则
- benchmark 标签逻辑

### 6.4 推荐接口

```python
class VerificationPolicy:
    def should_verify(self, state, candidate_result) -> bool: ...
    def build_verification_step(self, state, candidate_result) -> dict: ...
```

### 6.5 建议文件归属

建议文件：

- `terminus/policies/verification_policy.py`
- `terminus/prompts/verification/`
- `terminus/configs/verification/`

### 6.6 验证要求

必须通过：

- `smoke_terminal`
- `verification_smoke`
- `tb2_canary`

建议额外记录指标：

- verification-before-finish ratio
- false finish rate
- fix 后未验证直接结束比例

## 7. 模块四：Finish Policy

### 7.1 职责

`finish_policy` 负责判断何时结束任务，以及结束时给出什么 finish reason。

### 7.2 允许修改的内容

- should_finish 判断逻辑
- finish 前置条件
- finish reason 分类逻辑
- finish 阈值与保守性设置

### 7.3 禁止修改的内容

- 最终任务 evaluator
- reward 聚合逻辑
- 终止任务的系统级安全条件

### 7.4 推荐接口

```python
class FinishPolicy:
    def should_finish(self, state) -> bool: ...
    def finish_reason(self, state) -> str: ...
```

### 7.5 建议文件归属

建议文件：

- `terminus/policies/finish_policy.py`
- `terminus/prompts/finish/`
- `terminus/configs/finish/`

### 7.6 验证要求

必须通过：

- `smoke_terminal`
- `finish_decision_smoke`
- `tb2_canary`

建议额外记录指标：

- premature finish rate
- max-iteration avoidance rate
- finish 后回归失败比例

## 8. 模块五：Parallelization Policy

### 8.1 职责

`parallelization_policy` 负责判断是否需要 fanout，并决定可并发子任务如何拆分。

该模块是“让模型自主演化异步逻辑”的关键模块。

### 8.2 允许修改的内容

- 是否 fanout 的判断逻辑
- 哪些场景适合并发
- 子任务拆分规则
- read-only 子任务定义规则
- 并发前的信息收集触发条件

### 8.3 禁止修改的内容

- 底层 async executor
- 线程/协程调度底层实现
- 共享状态底层写路径
- 冲突检测底层机制

### 8.4 推荐接口

```python
class ParallelizationPolicy:
    def should_fanout(self, state, trace_summary) -> bool: ...
    def propose_subtasks(self, state, trace_summary) -> list[dict]: ...
    def choose_merge_policy(self, state, branch_results) -> str: ...
```

### 8.5 建议文件归属

建议文件：

- `terminus/policies/parallelization_policy.py`
- `terminus/prompts/parallelization/`
- `terminus/configs/parallelization/`

### 8.6 验证要求

必须通过：

- `smoke_terminal`
- `parallelization_smoke`
- `tb2_canary`

对于该模块，还必须额外满足：

- `shared_state_conflict_rate == 0`
- `branch_merge_failure_rate <= threshold`
- `wall_clock_latency` 不显著恶化

建议额外记录指标：

- fanout rate
- 平均 branch 数量
- branch success rate
- sync bottleneck 改善幅度

## 9. 模块六：Merge Policy

### 9.1 职责

`merge_policy` 负责将并发子任务结果合并回主状态。

### 9.2 允许修改的内容

- branch 结果摘要方式
- branch 结果优先级策略
- merge 输出格式
- merge 后回传 planner 的结构

### 9.3 禁止修改的内容

- 共享状态底层存储
- 冲突检测的底层实现
- 未注册对象的任意写入

### 9.4 推荐接口

```python
class MergePolicy:
    def merge(self, state, branch_results) -> dict: ...
```

### 9.5 建议文件归属

建议文件：

- `terminus/policies/merge_policy.py`
- `terminus/prompts/merge/`
- `terminus/configs/merge/`

### 9.6 验证要求

必须通过：

- `merge_smoke`
- `parallelization_smoke`
- `tb2_canary`

建议额外记录指标：

- merge failure rate
- branch result utilization rate
- merge 后错误规划比例

## 10. 模块间依赖关系

建议依赖关系如下：

- `planner_policy` 依赖 `context_summarizer` 和 `memory_policy`
- `retry_policy` 依赖 `planner_policy` 输出的当前 plan
- `verification_policy` 依赖 `finish_policy` 的 finish 前状态
- `parallelization_policy` 依赖 `planner_policy` 和 trace signals
- `merge_policy` 依赖 `parallelization_policy` 产出的 branch schema

原则上：

- 模型可以修改模块内部逻辑
- 模型不应随意改变模块间依赖方向
- 模型不应跨模块引入隐式共享状态

## 11. Patch 风险等级建议

建议为每个模块定义默认风险等级：

- `planner_policy`: medium
- `retry_policy`: medium
- `verification_policy`: medium
- `finish_policy`: high
- `parallelization_policy`: high
- `merge_policy`: high

原因：

- `finish_policy` 容易导致过早结束
- `parallelization_policy` 和 `merge_policy` 容易引入并发冲突或状态不一致

## 12. 第一阶段推荐开放顺序

不要一开始就全部开放。

建议顺序：

1. `planner_policy`
2. `retry_policy`
3. `verification_policy`
4. `finish_policy`
5. `parallelization_policy`
6. `merge_policy`

理由：

- planner / retry / verification 通常是提升最稳定的模块
- finish 改动风险更高，但收益也大
- parallelization / merge 最复杂，最好等 trace 和 state 管理稳定后再开放

## 13. Patch 提交格式要求

每个 patch candidate 应至少包含以下信息：

```json
{
  "target_modules": ["verification_policy"],
  "files": [
    {
      "path": "terminus/policies/verification_policy.py",
      "change_type": "modify",
      "intent": "require at least one lightweight verification before finish when code changes were made"
    }
  ],
  "risk": "medium",
  "required_tests": ["smoke_terminal", "verification_smoke", "tb2_canary"],
  "rollback_if": [
    "false_finish_rate increases",
    "tb2_score drops"
  ]
}
```

## 14. 成功标准

可以认为 patchable modules 设计完成的标准包括：

- 每个模块有清晰职责
- 每个模块有明确接口
- 每个模块有白名单文件
- 每个模块有测试要求
- 模型生成的 patch 可以按模块归类
- patch 失败时可以定位到具体模块

## 15. 最终建议

对于 Terminus 的可修改模块设计，最重要的不是一开始开放很多模块，而是：

- 先把高价值策略模块抽干净
- 把底层安全逻辑和评测逻辑冻结住
- 让模型在明确的接口和白名单里演化逻辑
- 通过模块级 smoke + canary 验证来决定是否 promote

这样后续不管是自演化异步逻辑，还是自演化 retry / verification / finish 逻辑，都会更稳定、更可控、更容易复用。
