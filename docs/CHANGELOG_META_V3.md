# Meta-Learning v3 Changelog — Agent Code Modification

## 概述

v3 实现了从"只能修改 prompt 参数"到"可以直接修改 agent 代码"的跨越。新增了两个全新的修改层次：代码级行为策略（behavior_policy）和 agent 模板文件重写（template_edits）。

---

## 新增修改层次

### 1. 代码级行为策略（behavior_policy）

**文件**: `behavior_policy.yaml` → `MetaBehaviorConfig` dataclass → Python 控制流

与 prompt-level 的"建议"不同，behavior_policy 的配置直接改变 terminus_2.py 中的 Python 代码路径：

| 功能 | 配置字段 | 代码效果 |
|------|----------|----------|
| 自反思 | `enable_self_reflection`, `reflect_every_n_steps`, `reflection_prompt` | 每 N 步在终端输出后追加反思提问 |
| 命令过滤 | `blocked_command_patterns` | 正则匹配的命令在执行前被拦截 |
| 命令截断 | `max_commands_per_step` | 超出限制的命令被物理截断 |
| 时长限制 | `max_single_duration_sec` | 单条命令执行时间硬上限 |
| 输出控制 | `max_output_bytes` | 控制 agent 看到的终端输出长度 |
| 空命令检测 | `max_consecutive_empty_commands` | 连续空命令后强制终止 |
| 解析错误检测 | `max_consecutive_parse_errors` | 连续解析失败后强制终止 |

### 2. Agent 模板文件重写（template_edits）

**文件**: LLM 生成完整模板内容 → TemplateValidator 验证 → 覆写 .txt 文件

可重写的模板文件：
- `terminus-json-plain.txt` / `terminus-xml-plain.txt` — 主 system prompt
- `timeout.txt` — 超时反馈
- `summarize-summary.txt` / `summarize-questions.txt` / `summarize-answers.txt` — 总结流程
- `completion-confirmation-json.txt` / `completion-confirmation-xml.txt` — 完成确认

---

## Harbor 代码改动

### terminus_2.py

1. **新增 `MetaBehaviorConfig` dataclass**: 定义了 10 个代码级行为钩子字段，通过 `from_overrides()` 从 meta_overrides dict 解析
2. **新增 `_validate_commands()` 方法**: 实现命令过滤（正则匹配）、截断（max_commands_per_step）、时长限制
3. **自反思检查点**: 在 `_run_agent_loop()` 中，每 N 步在终端输出后追加反思提问
4. **空命令/解析错误计数器**: 连续异常时自动终止，避免无限循环
5. **模板文件加载**: `__init__` 中加载 summarize-*.txt 和 completion-confirmation-*.txt
6. **硬编码 prompt 提取**: 将 `_summarize()` 中的 3 个硬编码 prompt 和 `_get_completion_confirmation_message()` 中的 2 个硬编码 prompt 提取为外部模板文件

### 新增模板文件

| 文件 | 来源 |
|------|------|
| `templates/summarize-summary.txt` | 从 `_summarize()` 中的 `summary_prompt` 提取 |
| `templates/summarize-questions.txt` | 从 `_summarize()` 中的 `question_prompt` 提取 |
| `templates/summarize-answers.txt` | 从 `_summarize()` 中的 `answer_request_prompt` 提取 |
| `templates/completion-confirmation-json.txt` | 从 `_get_completion_confirmation_message()` 提取 |
| `templates/completion-confirmation-xml.txt` | 从 `_get_completion_confirmation_message()` 提取 |

---

## SkyRL 代码改动

### template_validator.py（新增）

模板验证器，在写入前检查：
- 必需占位符是否存在
- Schema 关键词是否存在（如 JSON 模板中的 `"analysis"`, `"commands"` 等）
- 内容长度是否在合理范围内
- 占位符是否被意外重复

### patch_executor.py

- 新增 `template_base` 参数，支持两个写入区域（override_base + template_base）
- 新增 `_write_template()` 方法：验证 → 备份 → 写入
- `_OVERRIDE_SCHEMA` 新增 `behavior_policy` 的 10 个字段

### llm_patch_planner.py

- System prompt 更新：描述三级修改能力（YAML overrides + behavior_policy + template_edits）
- 新增 `template_base` 参数
- `_parse_response_to_candidate()` 支持解析 `template_edits` 字段
- 输出格式新增 `template_edits: {filename: content}` 字段
- 风险评估：behavior_policy → medium, template_edits → high

### patch_planner.py

- `_ALLOWED_PHASE1_MODULES` 扩展：新增 `system_prompt_overrides`, `strategy_library`, `behavior_policy`
- `_MODULE_OVERRIDE_FILE` 新增 `behavior_policy` 映射

### patchable_modules.py

新注册 4 个可修改模块：
- `behavior_policy` — 代码级行为钩子
- `system_prompt_template` — 主 system prompt 模板
- `summarization_templates` — 总结流程模板
- `completion_templates` — 完成确认模板

### terminus2_adapter.py

- 新增 `template_base` 属性
- `build_override_section()` 支持 `behavior_policy` 摘要渲染

### harbor_generator.py

- `MetaLoopConfig` 新增 `template_base` 字段
- `_MetaLoopController` 自动从 adapter 获取 `template_base`
- `PatchExecutor` 和 `LLMPatchPlanner` 初始化时传入 `template_base`
- `_load_override_content()` 支持加载 `behavior_policy.yaml`
