# Meta-Learning Toolkit v2 — 变更文档

本文档详细记录了 meta-learning toolkit 从 v1 到 v2 的所有代码变更，包括 bug 修复、新增功能模块和架构重构。

---

## 1. Wandb 奖励指标覆盖问题修复

**文件**: `skyrl/train/trainer.py` — `_meta_training_step`

**问题**: `_meta_training_step` 调用 `postprocess_generator_output` 处理 meta-RL 批次时，会将 meta-RL 的 `delta_score` 奖励（通常在 ±0.06 量级）写入 `self.all_metrics["reward/*"]`，覆盖了主任务批次已经计算好的 `reward/avg_raw_reward`、`reward/avg_pass_at_N` 等指标。这导致 Wandb 上显示的奖励值异常低。

**修复方案**:
- 在调用 `postprocess_generator_output` 之前，保存 `self.all_metrics` 中所有 `reward/` 前缀的指标
- 调用后，将 meta 批次产生的 `reward/*` 指标迁移到 `meta_rl/reward/*` 命名空间
- 恢复主任务批次的原始 `reward/*` 指标

**效果**: Wandb 上现在能正确显示两组独立指标：
- `reward/*` — 主任务奖励（如 `reward/avg_raw_reward: 0.4375`）
- `meta_rl/reward/*` — meta-RL 奖励（如 `meta_rl/reward/avg_raw_reward: 0.06`）

---

## 2. 多轮交互式诊断器（Interactive Diagnoser）

**新增文件**:
- `skyrl_agent/meta_toolkit/diagnosis/interactive_diagnoser.py`
- `skyrl_agent/meta_toolkit/diagnosis/diagnosis_tools.py`
- `skyrl_agent/meta_toolkit/diagnosis/diagnosis_history.py`

### 2.1 InteractiveDiagnoser

替代了原有的 `LLMDiagnoser` 单次调用模式。新的诊断器实现了多轮对话流程：

1. **初始化**: 向 LLM 提供批次统计摘要（成功率、失败分布、平均轮次等）和近期历史
2. **工具调用循环**: LLM 通过 JSON 格式调用诊断工具，每次调用返回结果后继续分析
3. **提交诊断**: LLM 在充分调查后提交结构化诊断结果
4. **回退**: 超过最大轮次（默认 8 轮）或异常时，回退到规则诊断器

### 2.2 诊断工具（Diagnosis Tools）

| 工具名 | 功能 |
|--------|------|
| `inspect_trace` | 查看特定 trace 的详细事件（tool_call、turn 内容等） |
| `compare_traces` | 并排对比两条 trace 的关键指标差异 |
| `get_failure_distribution` | 汇总当前批次的失败统计 |
| `get_history` | 查看最近 N 个 meta-learning 周期的历史 |
| `submit_diagnosis` | 提交最终诊断（结束会话） |

### 2.3 DiagnosisHistory

跨周期诊断历史模块，持久化到 JSONL 文件。每个周期记录：
- 诊断出的问题类型
- 生成/接受/拒绝的 patch 数量
- 最佳 delta_score
- 当前激活的 override 快照

为后续诊断提供上下文，避免重复尝试失败的 patch 策略。

---

## 3. 扩展修改范围 — System Prompt 与策略库

### 3.1 新增 Override 类型

| 文件名 | 用途 |
|--------|------|
| `system_prompt_overrides.yaml` | 高级策略指令，直接注入 agent prompt |
| `strategy_library.yaml` | 少样本策略库，提供 pattern→steps 映射 |

### 3.2 system_prompt_overrides — 自由文本策略指令

```yaml
strategy_hints:
  - "Before editing any file, first run the existing test suite to establish a baseline"
  - "When encountering import errors, check the project's dependency file before modifying code"
  - "If a test fails after your edit, read the FULL error traceback before attempting a fix"
```

与 v1 的固定字段参数不同，`strategy_hints` 是一个 `list[str]`，每个元素由 LLM planner 自由生成。这意味着 meta-learning 可以"发明"全新的行为指令，不受预定义字段的约束。

注入方式：以 `[STRATEGY DIRECTIVES]` 格式渲染为文本，注入到 agent 的 task instruction 前缀中。

### 3.3 strategy_library — 结构化少样本策略

```yaml
strategies:
  - pattern: "test fails with ImportError after code edit"
    steps:
      - "Read the full error traceback to identify the missing module"
      - "Search the project for existing imports of that module"
      - "Fix the import path or add the dependency"
      - "Re-run the failing test to verify"
  - pattern: "agent has made 3+ edits without test improvement"
    steps:
      - "Stop editing and re-read the original task instruction"
      - "Run git diff to review all changes"
      - "Revert to last known-good state if needed"
      - "Re-plan from scratch"
```

策略库以 `[LEARNED STRATEGIES]` 格式注入，为 agent 提供条件-动作对形式的操作模板。

### 3.4 能力边界说明

需要明确的是，v2 新增的两种 override 类型 **仍然是 prompt-level 的修改**，而非直接修改 agent 的 Python 源代码。所有内容最终都渲染为文本注入到 agent 的 instruction 前缀中，agent 是否遵循取决于底层 LLM 的 instruction following 能力。

v2 的进步在于：从"只能调 15 个预定义数值参数"扩展到"可以生成任意自然语言策略指令和结构化操作模板"，大幅提升了修改的表达力。

### 3.4 受影响的文件

- `skyrl_agent/meta_toolkit/editing/patch_executor.py` — `_OVERRIDE_SCHEMA` 增加新字段
- `skyrl_agent/meta_toolkit/editing/llm_patch_planner.py` — LLM prompt 增加新字段文档，`_MODULE_OVERRIDE_FILE` 增加新映射
- `skyrl_agent/meta_toolkit/editing/patch_planner.py` — `_MODULE_OVERRIDE_FILE` 增加新映射
- `harbor/agents/terminus_2/terminus_2.py` — `_build_meta_override_section` 支持 `[STRATEGY DIRECTIVES]` 和 `[LEARNED STRATEGIES]`

---

## 4. Agent 通用化 — MetaOverrideAdapter 抽象层

**新增文件**:
- `skyrl_agent/meta_toolkit/adapters/__init__.py`
- `skyrl_agent/meta_toolkit/adapters/base.py`
- `skyrl_agent/meta_toolkit/adapters/terminus2_adapter.py`
- `skyrl_agent/meta_toolkit/adapters/generic_instruction_adapter.py`

### 4.1 设计动机

v1 将 Terminus-2 的逻辑硬编码在 `harbor_generator.py` 中。要支持 Terminus-1、Claude Code、Aider 等 agent，需要一个抽象接口来隔离 agent 差异。

### 4.2 MetaOverrideAdapter 接口

```python
class MetaOverrideAdapter(ABC):
    def agent_name_pattern(self) -> str: ...
    def matches(self, agent_name: str) -> bool: ...
    def build_registry(self) -> ModuleRegistry: ...
    def build_override_section(self, overrides: dict) -> str: ...
    def inject_into_trial_config(self, config: dict, overrides: dict) -> None: ...
    def supported_override_files(self) -> list[str]: ...
```

### 4.3 已实现的 Adapter

| Adapter | 匹配规则 | 注入方式 |
|---------|----------|----------|
| `Terminus2Adapter` | `terminus(-[12])?` | 通过 `meta_overrides` kwarg 注入 |
| `GenericInstructionAdapter` | `.*`（通配兜底） | 通过 instruction 文本前缀注入 |

### 4.4 harbor_generator.py 的变更

`HarborGenerator.__init__` 不再检查 `agent_name.startswith("terminus")`，而是遍历 adapter 列表，选择第一个匹配的 adapter：

```python
adapters = [Terminus2Adapter(), GenericInstructionAdapter()]
for adapter in adapters:
    if adapter.matches(self._agent_name):
        self._meta_adapter = adapter
        break
```

`_MetaLoopController` 接收 `adapter` 参数，用于后续 override 注入。

---

## 5. 文件变更清单

### SkyRL 仓库

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `skyrl/train/trainer.py` | 修改 | 修复 reward 覆盖 bug |
| `skyrl_agent/meta_toolkit/__init__.py` | 修改 | 导出 `MetaOverrideAdapter` |
| `skyrl_agent/meta_toolkit/diagnosis/__init__.py` | 修改 | 导出新模块 |
| `skyrl_agent/meta_toolkit/diagnosis/interactive_diagnoser.py` | 新增 | 多轮交互式诊断器 |
| `skyrl_agent/meta_toolkit/diagnosis/diagnosis_tools.py` | 新增 | 诊断工具函数 |
| `skyrl_agent/meta_toolkit/diagnosis/diagnosis_history.py` | 新增 | 跨周期诊断历史 |
| `skyrl_agent/meta_toolkit/adapters/__init__.py` | 新增 | Adapter 模块 |
| `skyrl_agent/meta_toolkit/adapters/base.py` | 新增 | `MetaOverrideAdapter` 抽象基类 |
| `skyrl_agent/meta_toolkit/adapters/terminus2_adapter.py` | 新增 | Terminus-2 adapter |
| `skyrl_agent/meta_toolkit/adapters/generic_instruction_adapter.py` | 新增 | 通用 instruction 注入 adapter |
| `skyrl_agent/meta_toolkit/editing/patch_executor.py` | 修改 | 支持新 override 类型 |
| `skyrl_agent/meta_toolkit/editing/llm_patch_planner.py` | 修改 | LLM prompt 增加新字段 |
| `skyrl_agent/meta_toolkit/editing/patch_planner.py` | 修改 | 增加 override 文件映射 |
| `examples/.../harbor_generator.py` | 修改 | adapter 模式 + 诊断历史记录 |

### Harbor 仓库

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `src/harbor/agents/terminus_2/terminus_2.py` | 修改 | `_build_meta_override_section` 支持策略指令和策略库 |

---

## 6. 配置兼容性

所有变更向后兼容：
- 未配置 `llm_model` 时仍使用规则诊断器，行为不变
- 新 override 文件为可选项，不影响已有 override
- Adapter 自动匹配：Terminus 系列 agent 走 `Terminus2Adapter`，其他 agent 走 `GenericInstructionAdapter`
- 现有的 `default.yaml` 配置无需修改
