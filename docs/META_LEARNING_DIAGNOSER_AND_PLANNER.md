# 元学习诊断器与规划器

本文档介绍 SkyRL 元学习框架中诊断器（Diagnoser）和规划器（Planner）的工作原理。

## 1. 整体流程

元学习的核心是一个闭环：

```
训练轨迹 → 诊断器（发现问题）→ 规划器（生成补丁）→ 金丝雀评估 → 接受/拒绝
```

每隔 N 个训练 batch，系统会收集最近的智能体执行轨迹，交给诊断器分析失败原因，再由规划器生成具体的代码或配置修改方案。修改方案经过金丝雀评估验证后，如果效果有提升则被接受，否则回滚。

## 2. 诊断器（Diagnoser）

### 2.1 角色

诊断器的任务是从大量执行轨迹中找出智能体的系统性问题。它不知道自己在诊断哪个具体的智能体——它只看到轨迹数据（任务 ID、奖励、步数、失败原因、工具调用等），然后给出结构化的诊断结论。

### 2.2 工作方式

诊断器采用**多工作者并行诊断**的架构：

1. **任务分组**：首先将当前 batch 的所有轨迹按任务 ID 分组，然后将任务分为三类：
   - **全部失败**的任务（所有轨迹 reward=0）
   - **部分成功**的任务（有对有错，最有分析价值）
   - **全部成功**的任务

2. **并行工作者**：启动 3 个独立的诊断工作者，每个工作者关注不同的任务子集：
   - 工作者 1：聚焦全部失败的任务，分析为什么一个任务的所有尝试都失败了
   - 工作者 2：聚焦部分成功的任务，对比同一任务的成功和失败轨迹
   - 工作者 3：聚焦上下文长度相关问题，分析长轨迹的失败模式

3. **多轮交互**：每个工作者最多进行 16 轮与 LLM 的对话。工作者可以调用诊断工具来深入分析：
   - `get_failure_distribution`：查看失败类型的统计分布
   - `get_trace_detail`：查看某条具体轨迹的详细信息
   - `compare_traces`：对比两条轨迹的差异
   - `get_task_overview`：查看某个任务的所有轨迹概况
   - `check_history`：查看历史诊断记录，避免重复

4. **强制提交**：如果工作者在 16 轮内没有主动提交诊断，系统会在倒数第 2 轮注入强制提交提示。如果最终仍未提交，系统会从对话历史中提取关键发现，合成一份诊断报告。

5. **结果合并**：多个工作者的诊断结果被合并去重，保留每种问题类型中置信度最高的诊断。

### 2.3 诊断输出

每条诊断包含：
- **problem_type**：问题类型（见下表）
- **root_cause_hypotheses**：根因假设列表
- **candidate_modules**：建议修改的模块（如 planner_policy、behavior_policy、system_prompt_overrides 等）
- **confidence**：置信度（0-1）
- **affected_task_ids**：受影响的任务 ID
- **analysis_summary**：从诊断对话中提取的详细分析摘要

### 2.4 问题类型体系

诊断器使用精细化的问题类型，每种类型对应下游规划器的特定改进策略：

| 问题类型 | 典型证据 | 下游策略方向 |
|---|---|---|
| premature_completion | 低奖励 + 少步数，过早标记完成 | Smart Completion 验证 |
| output_overflow | 高步数 + 重复命令，上下文溢出 | 智能输出截断/过滤 |
| wasteful_waiting | 高耗时 / 步数比，快命令设长等待 | Duration 优化 |
| parse_error_loop | 高 tool_failures，反复解析失败 | 错误恢复提示 |
| stuck_in_loop | 相同命令重复执行，奖励无进展 | 进度追踪 + 策略切换 |
| planning_failure | 简单任务多步完成，无清晰策略 | 策略引导 |
| timeout | 大量超时 | 超时恢复 + Duration 限制 |
| verification_gap | 通过但低奖励，部分解决 | 后步验证 |
| recovery_failure | 错误后无法恢复 | 重试策略 |
| context_overload | Token 上限触发，摘要丢失信息 | 输出管理 |
| tool_exhaustion | Episode 耗尽 | 效率优化 |

## 3. 规划器（Planner）

### 3.1 角色

规划器接收诊断结果和失败轨迹样本，生成具体的修改方案（补丁）。与诊断器不同，规划器**知道智能体的代码结构**——它会看到智能体的实际 agent loop 代码，从而决定如何通过 YAML 配置或 Python hook 来改进智能体。

### 3.2 两种修改手段

规划器可以生成两类修改：

**Part A：YAML 配置覆盖（低风险）**

通过修改配置文件来调整智能体行为，包括：
- **planner_policy**：规划风格、子目标深度等
- **retry_policy**：重试次数、退避策略
- **verification_policy**：是否每步验证
- **finish_policy**：完成确认、最少步数要求
- **system_prompt_overrides**：注入策略提示到智能体的 prompt 中
- **strategy_library**：注入学习到的策略模式
- **behavior_policy**：命令限制、输出截断等

**Part B：代码 Hook（高影响）**

通过注入 Python 函数到智能体的执行循环中，在 5 个注入点修改行为：
- **before_llm_call**：在调用 LLM 前修改 prompt（可注入反思提示、上下文管理等）
- **before_execute**：在执行命令前修改命令列表（可过滤危险命令、添加验证步骤）
- **after_execute**：在执行后处理终端输出（可智能过滤、高亮错误）
- **on_timeout**：处理命令超时（可改进恢复策略）
- **on_parse_error**：处理 LLM 输出解析失败（可构造重试提示）

每个 hook 是一个名为 `hook` 的 Python 函数，运行在 try/except 中——如果 hook 出错，系统会使用原始值，不会影响智能体的正常运行。

### 3.3 每次补丁的限制

- 每个补丁最多 **3 处修改**（YAML 模块 + hook 合计）
- 必须基于诊断证据，不能凭空修改
- 配置值不能设为默认值（无意义的修改）
- hook 代码只能使用安全的标准库（re、json、math、collections 等）

### 3.4 改进策略引导（基于 KIRA 分析）

规划器的提示词中包含 8 种改进策略，每种策略对应诊断器可能发现的特定问题。这些策略不是硬编码的规则，而是**引导 LLM 思考方向的提示**——LLM 需要根据实际的诊断证据判断是否适用：

| 诊断问题 | 策略方向 | 实现方式 |
|---|---|---|
| premature_completion | Smart Completion 验证 | `before_llm_call` hook 注入多角度验证 checklist |
| output_overflow | 智能输出管理 | `after_execute` hook 过滤/摘要终端输出 |
| wasteful_waiting | Duration 优化 | `before_execute` hook 按命令类型调整等待时间 |
| parse_error_loop | 错误恢复 | `on_parse_error` hook 构造针对性重试提示 |
| stuck_in_loop | 进度追踪 | `before_llm_call` hook 检测重复命令并警告 |
| planning_failure | 策略引导 | `strategy_hints` + `planner_policy` 调整 |
| timeout | 超时恢复 | `on_timeout` hook 发送 Ctrl+C 并捕获部分输出 |
| verification_gap | 后步验证 | `verification_policy` + `strategy_hints` |

这些策略的灵感来源于 [Terminus-KIRA](https://github.com/krafton-ai/KIRA) 项目的改进分析，但通过 meta-learning 的方式让 LLM 自主决定何时、如何应用。

### 3.5 多样性生成

规划器会用同一个 prompt 调用 LLM 多次（默认 4 次，temperature=0.7），生成多个不同的补丁候选。每个候选独立经过金丝雀评估，只有效果最好且超过阈值的才会被接受。

## 4. 从诊断到补丁的信息流

```
诊断器输出:
  - problem_type: "Premature completion"
  - confidence: 0.85
  - root_cause: "Agent marks task complete without verifying output"
  - affected_tasks: ["task_42", "task_67", "task_103"]
  - analysis_summary: "In 15/20 failed traces, the agent set is_task_complete=true
    on the first or second turn without checking command output..."

        ↓ 传递给规划器

规划器收到:
  - 上述诊断结论
  - 10 条失败轨迹的摘要
  - 当前已激活的修改（避免重复）
  - 智能体的实际代码（terminus-1 或 terminus-2 的 agent loop）

        ↓ 规划器生成补丁

补丁示例 (Smart Completion):
  {
    "overrides": {
      "finish_policy": {
        "require_confirmation": true,
        "min_steps_before_finish": 5
      }
    },
    "code_hooks": {
      "before_llm_call": "def hook(prompt, context):\n    if context.is_task_complete and not context.kv.get('confirmed'):\n        context.kv['confirmed'] = True\n        prompt += '\\n[VERIFICATION] Before confirming completion, check:\\n- Does the solution meet all requirements?\\n- Have you verified from test engineer, QA, and user perspectives?\\n- Are there any leftover files or side effects?'\n    return prompt"
    },
    "rationale": "Agent completes too early in 15/20 failed traces;
     adding multi-perspective verification checklist"
  }
```

## 5. 安全机制

整个流程有多层安全保障：

1. **Hook 验证**：所有 hook 代码在加载前经过 AST 语法检查和导入白名单验证
2. **运行时隔离**：hook 运行在 try/except 中，失败时回退到原始值
3. **金丝雀评估**：每个补丁必须在小规模任务集上证明有效才会被接受
4. **Git 版本控制**：所有补丁通过独立的 Git 仓库管理，拒绝的补丁会被原子回滚
5. **变更限制**：每个补丁最多 3 处修改，防止过度改动
