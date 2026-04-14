# SkyRL Meta-Learning: Agent Self-Improvement through Closed-Loop Optimization

## 项目概述

本项目在 SkyRL 强化学习训练框架上构建了一套 **元学习（Meta-Learning）闭环系统**，使训练中的 AI Agent 能够自动发现执行中的问题、生成行为修改策略、验证修改效果，并将成功的策略持久化为 Agent 的长期能力提升。

核心思想是：**让 LLM 在 RL 训练过程中不仅学习解决任务，还学习如何改进自己的工作方式。**

```
┌─────────────────────────────────────────────────────────────┐
│                    SkyRL Training Loop                       │
│                                                              │
│   ┌──────────┐    ┌───────────┐    ┌──────────────┐         │
│   │ Generate  │───▶│  Reward   │───▶│  RL Update   │         │
│   │ Rollouts  │    │ Compute   │    │  (GRPO)      │         │
│   └──────────┘    └───────────┘    └──────────────┘         │
│        │                                    │                │
│        ▼                                    │                │
│   ┌─────────────────────────────────────┐   │                │
│   │      Meta-Learning Closed Loop       │   │                │
│   │                                      │   │                │
│   │  Traces ──▶ Diagnose ──▶ Plan ──▶   │   │                │
│   │  Patch ──▶ Canary ──▶ Promote       │   │                │
│   │           │                          │   │                │
│   │           ▼                          │   │                │
│   │     Meta-RL Samples ─────────────────┼───┘                │
│   └─────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 系统架构

### Pipeline 全景

每隔 N 个训练 batch，meta-learning 闭环执行以下六个阶段：

```
[1. Observe]  ──▶  [2. Diagnose]  ──▶  [3. Plan]
                                            │
[6. Promote]  ◀──  [5. Canary]   ◀──  [4. Patch]
      │
      ▼
[Agent Updated] ──▶ [Next Training Batch]
```

| 阶段 | 模块 | 输入 | 输出 |
|------|------|------|------|
| **Observe** | `TraceWriter` | Harbor Trial 结果 | `TraceRecord` 列表 |
| **Diagnose** | `InteractiveDiagnoser` / `RuleBasedDiagnoser` | TraceRecord 批次 | `DiagnosisResult` 列表 |
| **Plan** | `LLMPatchPlanner` / `PatchPlanner` | 诊断结果 + Traces | `PatchCandidate` 列表 |
| **Patch** | `PatchExecutor` | PatchCandidate | YAML Override + 行为策略 + 模板文件 + **Hook 代码** |
| **Canary** | `CanaryRunner` | Override 文件 + 任务集 | `PatchEvalResult`（含 delta_score）|
| **Promote** | `Promoter` + `PatchVersionControl` | Canary 评估结果 | Accept/Reject 决策 + Git 提交 |

### 与 SkyRL 训练循环的集成

```
SkyRL Trainer
    │
    ├── generate()  ──────────▶  HarborGenerator
    │                                 │
    │                                 ├── harbor_agent_loop() × N  ──▶  Harbor Trial
    │                                 │        │
    │                                 │        └── 收集 TraceRecord
    │                                 │
    │                                 ├── maybe_run_cycle()  ──────▶  Meta-Learning 闭环
    │                                 │        │
    │                                 │        └── 返回 MetaTrainingSample[]
    │                                 │
    │                                 └── 主批次 GeneratorOutput
    │
    ├── postprocess + RL update  ◀──  主批次
    │
    └── _meta_training_step()    ◀──  Meta-RL 批次（独立 mini-step）
```

---

## 核心模块详解

### 1. Observability — 轨迹记录

**文件**: `skyrl_agent/meta_toolkit/observability/`

每条 Agent 执行轨迹被结构化为 `TraceRecord`：

```python
@dataclass
class TraceRecord:
    task_id: str                    # 任务标识
    success: bool                   # 是否成功
    final_reward: float             # 最终奖励
    turn_count: int                 # 对话轮次
    finish_reason: str              # 结束原因 (complete/agent_timeout/context_length/error)
    failure_tags: list[str]         # 失败标签 (context_overload/sync_bottleneck/...)
    tool_calls: int                 # 工具调用次数
    tool_failures: int              # 工具失败次数
    events: list[TraceEvent]        # 逐轮事件序列
```

`TraceWriter` 将轨迹以 JSONL 格式持久化到磁盘，按 run_name 组织文件。

### 2. Diagnosis — 问题诊断

**文件**: `skyrl_agent/meta_toolkit/diagnosis/`

#### 2.1 多轮交互式诊断（InteractiveDiagnoser）

这是诊断系统的核心创新。与传统的 "dump all data → single LLM call" 不同，交互式诊断器给予 LLM **主动调查**的能力：

```
LLM: 我先看一下整体失败分布
     → {"tool": "get_failure_distribution"}

系统: Total: 32, Success: 14 (44%), Fail: 18
      Finish reasons: {"agent_timeout": 8, "context_length": 6, "complete": 18}
      Failure tags: {"context_overload": 10, "sync_bottleneck": 8}

LLM: context_overload 和 sync_bottleneck 数量差不多，让我对比一个成功和失败的 trace
     → {"tool": "compare_traces", "task_id_a": "task1-traj0", "task_id_b": "task2-traj0"}

系统: (并排对比表格)
      turns:         12 |  45       <--DIFF
      tool_calls:     8 |  38       <--DIFF
      finish_reason: complete | agent_timeout  <--DIFF

LLM: 失败 trace 的 tool_calls 远高于成功的，让我看看具体事件
     → {"tool": "inspect_trace", "task_id": "task2-traj0"}

系统: (前 30 个事件的详细内容)

LLM: 看到了，失败 trace 在第 20 轮之后进入了重复修改-测试循环。
     再看看历史上的 patch 有没有解决过类似问题
     → {"tool": "get_history", "n_cycles": 5}

系统: - Cycle 15: diagnosed=['context_overload'], promoted=1, best_delta=0.062
      - Cycle 16: diagnosed=['context_overload'], promoted=0, best_delta=-0.031

LLM: 上一个周期的 context_overload patch 效果不好，需要换个角度。
     提交诊断：
     → {"tool": "submit_diagnosis", "diagnoses": [
         {"problem_type": "tool_exhaustion",
          "root_cause_hypotheses": ["Agent enters edit-test loops >20 turns without progress"],
          "candidate_modules": ["planner_policy", "finish_policy"],
          "confidence": 0.75,
          "affected_task_ids": ["task2-traj0", "task5-traj1"]}
       ]}
```

**关键优势**:
- **证据驱动**: 诊断结论基于实际 trace 数据，而非简单统计
- **对比式分析**: 对比成功/失败案例发现关键差异
- **历史感知**: 避免重复尝试效果不好的 patch
- **可控深度**: 通过 `max_turns` 控制调查深度

#### 2.2 规则诊断器（RuleBasedDiagnoser）

在未配置 LLM 时自动使用。基于预定义规则从 TraceRecord 统计信息中匹配失败模式。

### 3. Planning — Patch 规划

**文件**: `skyrl_agent/meta_toolkit/editing/`

#### 3.1 LLM Patch Planner

接收诊断结果和失败 traces，生成三类修改：YAML 参数覆盖、行为策略配置、模板文件重写。核心特点：

- **多样性采样**: 同一 prompt 调用 N 次（temperature=0.7），生成 N 个不同的 `PatchCandidate`
- **GRPO 兼容**: N 个 candidates 共享一个 `instance_id`，形成 GRPO 组。每个 candidate 获得不同的 canary reward，使 RL 训练能区分好的和坏的规划策略
- **四级输出**: 一个 PatchCandidate 可以同时包含 YAML overrides、behavior_policy 配置、template_edits 和 **code_hooks**

#### 3.2 Agent 代码上下文注入

为了让 Patch Planner 的 LLM 能写出有效的 hook 和模板修改，system prompt 中包含了 Terminus-2 的**精选代码上下文**（约 80 行），涵盖：

- `Command` 数据类定义（`.keystrokes`, `.duration_sec`）
- Agent loop 简化流程图（标注了 5 个 hook 注入点）
- `_execute_commands` 串行执行逻辑
- `_handle_llm_interaction` 解析流程
- `_limit_output_length` 输出截断机制
- 关键观察提示（串行执行、tmux 窗口尺寸、context.kv 持久化等）

此外，user message 中还会自动附带**当前已激活的 overrides 和 hooks**（如果有的话），避免模型重复生成已有的修改。

#### 3.3 可修改范围的六个层次

| 层次 | 修改方式 | 风险 | 效果确定性 | 表达力 |
|------|----------|------|-----------|--------|
| **1. 数值参数** | YAML → prompt 文本 | 低 | 依赖 instruction following | 15 个固定字段 |
| **2. 策略指令** | `strategy_hints: list[str]` → prompt | 低 | 依赖 instruction following | 自由文本 |
| **3. 策略库** | `strategies: list[{pattern, steps}]` → prompt | 低 | 依赖 instruction following | 条件-动作对 |
| **4. 行为策略** | YAML → `MetaBehaviorConfig` → Python 控制流 | 中 | **确定性**（硬限制） | 10+ 代码钩子 |
| **5. 模板重写** | 直接覆写 agent 模板 .txt 文件 | 高 | **确定性**（改变 system prompt） | 重写整个 prompt |
| **6. Code Hooks** | 模型生成 Python 函数注入 agent 执行管线 | 高 | **确定性**（自定义逻辑） | **图灵完备** |

**层 1-3** 是 prompt-level 注入，LLM 自行决定是否遵循。**层 4-5** 是预定义钩子和模板修改。**层 6** 是本项目的核心创新——让模型直接编写 Python 代码来改变 agent 行为：

- **层 4（behavior_policy）**: 通过 `MetaBehaviorConfig` dataclass 控制 terminus_2.py 中的 `if/for/while` 控制流。例如 `max_commands_per_step: 5` 会物理截断命令列表，`blocked_command_patterns` 用正则在命令到达终端前拦截。
- **层 5（template_edits）**: 直接重写 agent 的 system prompt 模板等 8 个 .txt 文件。写入前经 `TemplateValidator` 验证占位符完整性，自动备份原文件，必须通过 canary A/B 测试才能 promote。
- **层 6（code_hooks）**: 模型生成 Python 函数，注入 agent 的 5 个执行管线节点。详见下一节。

#### 3.3 Code Hooks — 模型生成的 Python 代码注入

这是本项目最核心的能力：**让训练中的 LLM 不仅可以修改 prompt 参数，还可以编写 Python 代码直接改变 agent 的执行逻辑**。

**核心理念**: Terminus-2 的执行循环有 5 个关键节点（命令执行前、执行后、超时处理、解析错误处理、LLM 调用前），模型可以在这些节点注入自定义逻辑。

##### Hook 注入点

| Hook 点 | 签名 | 调用时机 | 示例用途 |
|---------|------|---------|---------|
| `before_execute` | `hook(commands, context) → commands` | 命令列表发送到终端前 | 合并短命令为管道、过滤危险命令、重排序 |
| `after_execute` | `hook(terminal_output, context) → str` | 终端输出返回给 LLM 前 | 提取关键信息、压缩输出、添加分析注释 |
| `on_timeout` | `hook(cmd, output, context) → str` | 命令超时后 | 自定义超时恢复策略 |
| `on_parse_error` | `hook(response, error, context) → str∣None` | LLM 输出解析失败时 | 自定义错误恢复提示 |
| `before_llm_call` | `hook(prompt, context) → str` | Prompt 发送给 LLM 前 | 动态注入上下文、条件性修改指令 |

##### Hook 安全机制

```
模型生成 hook 源码
    │
    ▼
[1] AST 静态分析  ──── 拒绝危险 import (os, subprocess, sys, ...)
    │                    只允许: re, json, math, collections, itertools, functools, copy
    ▼
[2] 语法编译验证  ──── 必须包含且仅包含一个名为 'hook' 的函数
    │
    ▼
[3] try/except 运行时包装 ──── hook 异常不影响 agent 正常运行
    │                          异常时返回原始值（安全降级）
    ▼
[4] HookContext 隔离  ──── hook 只能访问 HookContext 对象
    │                       不接触 agent 内部状态、session、network
    ▼
[5] Git 版本控制  ──── 所有 hook 文件纳入 meta_patches 的 git 仓库
                       canary A/B 测试通过后才 promote
                       任何时候可以 rollback 到上一个版本
```

##### HookContext — Hook 的受限 API

Hook 函数通过 `context` 参数获取信息，不能直接访问 agent 对象：

```python
@dataclass
class HookContext:
    episode: int = 0              # 当前 episode
    total_episodes: int = 0       # 已执行 episode 总数
    n_commands_executed: int = 0   # 已执行命令总数
    n_parse_errors: int = 0       # 累计解析错误数
    n_timeouts: int = 0           # 累计超时数
    last_analysis: str = ""       # LLM 最近的分析文本
    last_plan: str = ""           # LLM 最近的计划文本
    is_task_complete: bool = False # LLM 是否标记任务完成
    original_instruction: str = "" # 原始任务指令
    kv: dict = field(...)         # 持久化键值对（跨步骤共享状态）
```

##### 示例：模型发现命令执行效率低，生成管道合并 hook

诊断阶段 LLM 发现失败 trace 中 agent 逐条执行短命令（如 `cd dir`, `ls`, `cat file`），每条都等待终端响应，导致 episode 浪费。于是生成如下 hook：

```python
# before_execute hook: 将连续短命令合并为 shell 管道
def hook(commands, context):
    merged = []
    chain = []
    for cmd in commands:
        ks = cmd.keystrokes.rstrip("\n")
        # 短时间命令且无管道符号的，可以合并
        if cmd.duration_sec < 2.0 and "&&" not in ks and "|" not in ks:
            chain.append(ks)
        else:
            if chain:
                from copy import copy
                m = copy(commands[0])
                m.keystrokes = " && ".join(chain) + "\n"
                m.duration_sec = max(c.duration_sec for c in commands[:len(chain)])
                merged.append(m)
                chain = []
            merged.append(cmd)
    if chain:
        from copy import copy
        m = copy(commands[0])
        m.keystrokes = " && ".join(chain) + "\n"
        m.duration_sec = 2.0
        merged.append(m)
    return merged or commands
```

##### 示例：模型发现 agent 反复查看相同文件，生成上下文缓存 hook

```python
# after_execute hook: 在终端输出前追加已见输出的摘要
import json

def hook(terminal_output, context):
    seen = context.kv.setdefault("seen_outputs", {})
    import re
    # 提取文件路径（简单启发式）
    paths = re.findall(r'(?:cat|less|head|tail)\s+(\S+)', terminal_output)
    for p in paths:
        if p in seen:
            seen[p] += 1
        else:
            seen[p] = 1
    
    repeated = [p for p, c in seen.items() if c > 2]
    if repeated:
        hint = f"\n[META-HOOK] You have viewed these files {3}+ times: {', '.join(repeated)}. Consider a different approach.\n"
        return terminal_output + hint
    return terminal_output
```

这些 hook 由 LLM Patch Planner 在诊断结果基础上自动生成，经过语法验证后写入 `meta_patches/terminus2/hooks/` 目录，通过 canary A/B 测试后被 promote。

### 4. Validation — Canary 评估

**文件**: `skyrl_agent/meta_toolkit/validation/`

Canary 评估实现了 A/B 测试：

```
            ┌──── Task Set (4 tasks) ────┐
            │                            │
     Baseline Trials              Candidate Trials
    (current overrides)          (new patch applied)
            │                            │
     baseline_scores              candidate_scores
            │                            │
            └──── ComparisonEngine ──────┘
                        │
                   PatchEvalResult
                  (delta_score, regression)
```

**效率优化**: Baseline 只运行一次，所有 candidate 共享同一 baseline 结果。

**任务选取策略**: 优先选择诊断指出的失败任务，保留 1 个成功任务用于回归检测。

### 5. Promotion — 版本控制

**文件**: `skyrl_agent/meta_toolkit/runtime/`

- `Promoter` 根据 delta_score 和回归检测决定 accept/reject
- `PatchVersionControl` 在 override 目录维护独立 Git 仓库
- 每次接受的 patch 作为一次 commit，可以 rollback 到任意历史版本

```
meta_patches/terminus2/
├── .git/                    ← 独立 Git 仓库
├── planner_overrides.yaml
├── retry_overrides.yaml
├── verification_overrides.yaml
├── finish_overrides.yaml
├── system_prompt_overrides.yaml   ← (v2 新增)
├── strategy_library.yaml          ← (v2 新增)
├── behavior_policy.yaml           ← (v3 新增 — 代码级行为控制)
└── hooks/                         ← (v4 新增 — 模型生成的 Python hook)
    ├── before_execute.py          #   命令预处理
    ├── after_execute.py           #   输出后处理
    ├── on_timeout.py              #   超时恢复
    ├── on_parse_error.py          #   解析错误恢复
    └── before_llm_call.py         #   Prompt 动态修改
```

Agent 模板文件也受 Git 版本控制（在 Harbor 仓库中）：
```
harbor/src/harbor/agents/terminus_2/templates/
├── terminus-json-plain.txt              ← 可被 meta-learning 重写
├── terminus-xml-plain.txt               ← 可被 meta-learning 重写
├── timeout.txt                          ← 可被 meta-learning 重写
├── summarize-summary.txt                ← (v3 新增，从硬编码提取)
├── summarize-questions.txt              ← (v3 新增，从硬编码提取)
├── summarize-answers.txt                ← (v3 新增，从硬编码提取)
├── completion-confirmation-json.txt     ← (v3 新增，从硬编码提取)
└── completion-confirmation-xml.txt      ← (v3 新增，从硬编码提取)
```

### 6. Meta-RL Training — 元强化学习

Meta-learning 不只是一个离线优化循环，它的 LLM 推理过程（诊断 + 规划）本身也参与 RL 训练：

```
LLM 的诊断/规划对话
    ↓ 记录为 RecordedConversation
    ↓
构建 MetaTrainingSample
    ↓ reward = canary delta_score
    ↓
Tokenize → _pending_meta_batch
    ↓
Trainer._meta_training_step()
    ↓ 独立的 mini RL step (GRPO)
    ↓
LLM 学会产出更好的诊断和 patch 规划
```

**GRPO 分组**: 同一周期的 N 个 planning 对话共享同一 prompt，各自获得不同的 canary reward。GRPO 算法利用组内奖励差异计算相对优势，使模型偏向产出奖励更高的规划方案。

---

## Agent 通用化架构

### MetaOverrideAdapter 抽象

```python
class MetaOverrideAdapter(ABC):
    def agent_name_pattern(self) -> str: ...
    def build_registry(self) -> ModuleRegistry: ...
    def build_override_section(self, overrides: dict) -> str: ...
    def inject_into_trial_config(self, config: dict, overrides: dict) -> None: ...
```

### 已支持的 Agent 类型

| Adapter | 匹配 Agent | 注入方式 |
|---------|-----------|----------|
| `Terminus2Adapter` | terminus, terminus-1, terminus-2 | `meta_overrides` 构造器参数 → prompt 前缀 |
| `GenericInstructionAdapter` | 任意 agent（兜底） | Task instruction 文本前缀 |

### 扩展新 Agent

要为新 agent 添加 meta-learning 支持，只需实现一个 Adapter：

```python
class MyAgentAdapter(MetaOverrideAdapter):
    def agent_name_pattern(self) -> str:
        return r"my-agent.*"

    def build_registry(self) -> ModuleRegistry:
        registry = ModuleRegistry()
        registry.register(PatchableModule(
            name="planner_policy",
            description="...",
            ...
        ))
        return registry

    def build_override_section(self, overrides: dict) -> str:
        # 将 overrides 转换为你的 agent 能理解的文本格式
        ...

    def inject_into_trial_config(self, config: dict, overrides: dict) -> None:
        # 将 overrides 注入到 trial 配置中
        ...
```

然后在 `HarborGenerator.__init__` 的 adapter 列表中注册即可。

---

## 代码结构

```
skyrl-agent/skyrl_agent/meta_toolkit/
├── __init__.py                      # 顶层导出
├── llm_client.py                    # OpenAI 兼容 LLM 客户端 + 对话录制
│
├── adapters/                        # Agent 适配层
│   ├── base.py                      #   MetaOverrideAdapter 抽象接口
│   ├── terminus2_adapter.py         #   Terminus-2 适配器
│   └── generic_instruction_adapter.py  # 通用指令注入适配器
│
├── diagnosis/                       # 问题诊断
│   ├── interactive_diagnoser.py     #   多轮交互式 LLM 诊断器
│   ├── llm_diagnoser.py            #   单轮 LLM 诊断器（兼容保留）
│   ├── rule_based_diagnoser.py     #   规则诊断器
│   ├── diagnosis_tools.py          #   诊断工具函数
│   └── diagnosis_history.py        #   跨周期历史记录
│
├── editing/                         # Patch 生成与执行
│   ├── llm_patch_planner.py        #   LLM Patch 规划器（支持 overrides + behavior_policy + template_edits）
│   ├── patch_planner.py            #   规则 Patch 规划器
│   ├── patch_executor.py           #   Patch 执行器（写 YAML + 验证并写入模板文件）
│   ├── template_validator.py       #   模板验证器（占位符/schema/长度检查）
│   ├── module_registry.py          #   可修改模块注册表
│   └── patch_schema.py             #   PatchCandidate / PatchFileEdit
│
├── validation/                      # Canary 评估
│   ├── canary_runner.py            #   A/B 测试运行器
│   ├── comparison.py               #   统计比较引擎
│   └── patch_eval_result.py        #   评估结果数据类
│
├── runtime/                         # 运行时管理
│   ├── meta_controller.py          #   元控制器（失败标签汇总）
│   ├── promoter.py                 #   Patch 接受/拒绝决策
│   ├── patch_version_control.py    #   Git 版本控制
│   └── capability_runtime.py       #   能力运行时
│
├── hooks/                           # v4 新增: 模型生成代码注入
│   ├── __init__.py                 #   导出 HookContext, HookExecutor, HookPoint
│   ├── hook_context.py             #   HookContext — hook 的受限 API 表面
│   └── hook_executor.py            #   HookExecutor — 加载、验证、安全执行 hook
│
├── terminus/                        # Terminus 专用集成
│   ├── patchable_modules.py        #   Terminus 模块注册表构建（含 5 个 hook 模块）
│   ├── harbor_bridge.py            #   Harbor Trial → TraceRecord 转换
│   └── adapter.py                  #   Terminus 运行时适配
│
└── observability/                   # 轨迹观测
    ├── trace_schema.py             #   TraceRecord / TraceEvent 定义
    └── trace_writer.py             #   JSONL 持久化写入器
```

---

## 配置参考

在 SkyRL 训练脚本中通过 Hydra 配置元学习参数：

```bash
python main_harbor.py \
    generator.meta.enabled=true \
    generator.meta.interval_batches=1 \
    generator.meta.max_candidates=4 \
    generator.meta.canary_num_tasks=4 \
    generator.meta.canary_n_samples=3 \
    generator.meta.llm_model=Qwen3.5-4B \
    generator.meta.llm_base_url=http://localhost:8000/v1
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `meta.enabled` | `true` | 是否启用 meta-learning |
| `meta.interval_batches` | `20` | 每 N 个 batch 执行一次 meta 周期 |
| `meta.max_candidates` | `2` | 每个周期最大 patch 候选数 |
| `meta.canary_num_tasks` | `4` | Canary 评估任务数 |
| `meta.canary_n_samples` | `3` | 每个任务每侧的采样次数 |
| `meta.override_base` | `meta_patches/terminus2` | Override 文件存储路径 |
| `meta.llm_model` | `""` | 元推理 LLM 模型名（空=规则模式） |
| `meta.llm_base_url` | `http://localhost:8000/v1` | LLM API 地址 |

---

## 数据流图

```
                 Harbor Trial × N
                       │
                 ┌─────▼─────┐
                 │ TraceRecord│ × N
                 └─────┬─────┘
                       │
           ┌───────────▼───────────┐
           │   InteractiveDiagnoser│
           │   ┌─────────────────┐ │
           │   │ get_failure_dist│ │
           │   │ inspect_trace   │ │  ← 多轮工具调用
           │   │ compare_traces  │ │
           │   │ get_history     │ │
           │   └─────────────────┘ │
           └───────────┬───────────┘
                       │
                 DiagnosisResult[]
                       │
           ┌───────────▼───────────┐
           │   LLMPatchPlanner     │
           │   (N parallel calls)  │  ← temperature=0.7 → 多样性
           └───────────┬───────────┘
                       │
                 PatchCandidate[] (N个)
                       │
           ┌───────────▼───────────┐
           │   PatchExecutor       │
           │   写入 YAML Override  │
           │   写入 behavior_policy│
           │   验证+写入模板文件   │
           │   验证+写入 Hook .py  │  ← v4 新增
           └───────────┬───────────┘
                       │
              override files + templates + hooks on disk
                       │
           ┌───────────▼───────────┐
           │   CanaryRunner        │
           │   baseline (1次)      │
           │   candidates (N次)    │  ← A/B 测试
           └───────────┬───────────┘
                       │
                delta_score per candidate
                       │
           ┌───────────▼───────────┐
           │   Promoter            │
           │   accept / reject     │
           │   git commit          │  ← 版本控制
           └───────────┬───────────┘
                       │
              ┌────────┴────────┐
              │                 │
        Accept: 更新       Reject: 回滚
        _ACTIVE_OVERRIDES   （无操作）
              │
              ▼
        下一批训练自动使用新策略
```

---

## 关键设计决策

### 六层修改架构的设计理念

v4 实现了从"改配置/改 prompt"到"**模型直接编写 Python 代码改变 agent 行为**"的核心跨越。设计原则是 **渐进式风险**：

1. **层 1-3（prompt-level）**: 最安全，不可能导致崩溃，但效果依赖 LLM 的 instruction following
2. **层 4（behavior_policy）**: 通过 Python dataclass 控制代码路径，效果确定但范围受限于预定义字段
3. **层 5（template_edits）**: 可以重写整个 system prompt，有 TemplateValidator 保护
4. **层 6（code_hooks）**: **图灵完备** — 模型生成任意 Python 函数注入 agent 管线

每一层都有独立的安全机制：
- 层 1-3: YAML schema 验证
- 层 4: dataclass 类型检查 + 合理默认值
- 层 5: TemplateValidator（占位符完整性 + schema 关键词 + 长度边界）+ 自动备份 + canary eval
- 层 6: AST 静态分析（白名单 import）+ 语法编译验证 + try/except 运行时隔离 + HookContext API 限制 + Git 版本控制

### 为什么不直接在主 RL 批次中混入 Meta 样本？

Meta-RL 样本的奖励分布（canary delta_score，通常在 [-0.1, +0.2]）与主任务奖励（通常在 [0, 1]）差异很大。混合训练会干扰优势估计。因此使用独立的 `_meta_training_step` 进行训练。

### 为什么诊断需要多轮而不是一次性？

单次 LLM 调用处理 30+ 条 trace 的全部信息时，容易产出浅层、重复的诊断（如反复输出 "context_overload"）。多轮交互让 LLM 逐步缩小范围、对比案例，产出更具体且可操作的诊断。

---

## 与 Harbor 的关系

本项目依赖 Harbor 框架提供：

- **Agent 执行环境**: Docker 容器化的 Trial 执行
- **Agent 实现**: Terminus-2（及其他 Installed Agents）
- **验证系统**: test.sh → reward 文件
- **Override 注入点**: `_build_meta_override_section` 在 terminus_2.py 中

Harbor 的改动（v3 + v4）：

| 文件 | 改动 |
|------|------|
| `terminus_2.py` | v3: `MetaBehaviorConfig`, `_validate_commands()`, 自反思, 模板加载 |
| `terminus_2.py` | v4: 新增 `meta_hooks`/`meta_hooks_dir` 参数, `_run_hook()` 辅助方法, 5 个 hook 调用点 |
| `terminus_2.py` | 将硬编码 prompts 提取为外部模板文件 |
| `templates/summarize-*.txt` | 新增 3 个总结模板文件 |
| `templates/completion-confirmation-*.txt` | 新增 2 个完成确认模板文件 |

---

## 调试与日志

### 日志文件位置

| 日志 | 路径 | 内容 |
|------|------|------|
| **主训练日志** | `~/train.log` 或 stdout | 包含 `Meta:` 前缀的元学习状态信息 |
| **Meta LLM 对话记录** | `{log_dir}/meta_conversations.jsonl` | 每条 JSON 包含: cycle, role, messages, response, reward |
| **诊断历史** | `{log_dir}/diagnosis_history.jsonl` | 跨周期诊断摘要 |
| **Patch 版本历史** | `meta_patches/terminus2/.git/` | `git log` 查看所有 patch 提交 |
| **Harbor Trial 日志** | `{trials_dir}/` | 每个 trial 的 agent 轨迹 |
| **基础设施日志** | `/tmp/skyrl-logs/{run_name}/infra-*.log` | Ray/Megatron 底层日志 |

### 查看 Meta-Learning LLM 输出

```bash
# 查看最近的 LLM 对话（诊断 + 规划）
tail -5 /tmp/skyrl-logs/meta_conversations.jsonl | python -m json.tool

# 查看诊断对话
cat /tmp/skyrl-logs/meta_conversations.jsonl | python -c "
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    if r['role'] == 'diagnosis':
        print(f'Cycle {r[\"cycle\"]}: {r[\"response\"][:200]}')
"

# 查看规划对话及奖励
cat /tmp/skyrl-logs/meta_conversations.jsonl | python -c "
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    if r['role'] == 'planning':
        print(f'Cycle {r[\"cycle\"]} reward={r[\"reward\"]}: {r[\"response\"][:200]}')
"

# 查看 patch 版本历史
cd ~/SkyRL/examples/train_integrations/harbor/meta_patches/terminus2
git log --oneline -20

# 查看当前活跃的 hooks
ls -la hooks/
cat hooks/*.py
```

### 在训练日志中查找 Meta 相关信息

```bash
# 在 train.log 中搜索 meta 相关日志
grep "Meta:" ~/train.log | tail -30

# 查看 hook 加载信息
grep -i "hook" ~/train.log | tail -10
```

---

## 未来方向

- [ ] 支持更多 Agent 类型的 Adapter 实现
- [ ] 策略库的自动精简和合并
- [ ] 诊断工具扩展（性能 profiling、资源使用分析）
- [ ] Multi-objective meta-RL（同时优化成功率、效率、token 使用量）
- [ ] Hook 进化：跨周期积累 hook，自动组合多个 hook 的效果
- [ ] 模板进化：跨多个 canary 周期逐步优化 system prompt
- [ ] 安全沙箱：在隔离环境中测试高风险 hook 变更
- [ ] Hook 组合优化：自动检测多个 hook 之间的冲突和协同效应
