# Meta Hook 审计报告：问题 Hook 分析

## 一、核心发现：constraints-scheduling 为什么 168 轮停不下来

### 1.1 死亡循环的精确机制

以 `constraints-scheduling` 任务为例，agent 在第 5-6 轮就已经正确完成任务（创建了 ICS 会议文件），但之后陷入了 **168 轮死循环**直到 900s 超时。根本原因链如下：

**第一环：`[GUIDANCE]` hook 让 agent 忘记了 `task_complete` 字段**

文件 `before_llm_call_..._0bbc3050.py` 第 7 行：
```python
prompt += "\n[GUIDANCE] You must output valid JSON with exactly these fields: 'analysis', 'plan', 'commands'. Keep responses concise. Avoid verification loops."
```

这告诉 agent JSON 只需要 **3 个字段**：`analysis`、`plan`、`commands`。但 Terminus2 框架需要 `task_complete: true` 来结束任务！agent 于是每轮都输出：
```json
{
  "analysis": "The meeting_scheduled.ics file has been successfully created...",
  "plan": "Task is complete.",
  "commands": []
}
```
注意：**没有 `task_complete: true`**！agent 用自然语言说"Task is complete"但框架不认。

**第二环：`[CONTEXT CONSTRAINT]` hook 用策略文本替换了任务描述**

文件 `before_llm_call_..._d2d59c8d.py` 第 8-10 行：
```python
if call_count >= 5 and context.original_instruction:
    orig = context.original_instruction.strip()[:150]
    prompt += f"\n\n[CONTEXT CONSTRAINT] You are approaching context limits ({call_count} calls).\n"
    prompt += f"Task: {orig}\n"
```

`context.original_instruction` 的内容**以策略库开头**（`[LEARNED STRATEGIES] When: When creating a tar.gz archive...`），截取前 150 字后变成：
```
Task: [LEARNED STRATEGIES]
  When: When creating a tar.gz archive of /home/user/data/ directory, ensure the archive contains files at root level (alpha.txt,
```
这完全不是真实任务（真实任务是日程安排）！**每一轮 agent 看到的"任务描述"都是 tar 打包相关的策略文本**。

**第三环：多个 `after_round` hook 竞相 `request_new_turn`**

由于 agent 从未真正发出 `task_complete: true`，8 个 `after_round` hook 中多个检测到"未完成"→ 触发新轮次：
- `3151307f`：首次完成尝试 → `request_new_turn: True`
- `b1215fca`：首次完成 → `request_new_turn: True` + `[VERIFY BEFORE COMPLETE]`
- `c4d0b384`：验证循环检测 → `request_new_turn: True`
- `1fab62c7`：首次完成 → `request_new_turn: True`
- `5943530c`：验证计数 ≥ 3 → `request_new_turn: True`

### 1.2 Agent 视角的实际表现

| Episode | Agent 行为 | 系统响应 |
|---------|-----------|---------|
| 5 | 创建 ICS 文件，输出 `"commands": []` | `[GUIDANCE]` 注入（无 task_complete 字段） |
| 6 | "The task is complete"，空 commands | `[CONTEXT CONSTRAINT] Task: [LEARNED STRATEGIES] tar.gz...` |
| 7 | "I should mark the task as complete" | 同上，继续注入策略文本当任务描述 |
| 10 | "The warnings are about learned strategies..." | agent 意识到策略文本不对但无能为力 |
| 20 | "The user is stuck in a loop" | 继续 |
| 50 | "The task has been completed" | 继续 |
| 100 | "I need to keep my responses concise" | 继续 |
| 168 | 超时 | `AgentTimeoutError` 900s |

**Agent 从第 7 轮就知道任务完成了，但因为 hook 的干扰，168 轮都无法正确退出。**

---

## 二、问题 Hook 分类总览

共 37 个 Hook，按问题严重程度分为 4 级：

| 严重级别 | Hook 数量 | 描述 |
|---------|----------|------|
| **致命 (P0)** | 6 | 直接导致任务无法完成或死循环 |
| **严重 (P1)** | 8 | 严重干扰 agent 决策，高概率导致失败 |
| **中等 (P2)** | 10 | 增加噪声或浪费 episode，可能导致超时 |
| **低 (P3)** | 13 | 有小问题但影响有限 |

---

## 三、致命问题 Hook (P0)

### P0-1: `before_llm_call_..._0bbc3050.py` — 删除 task_complete 字段

**类型**: `before_llm_call`

**问题代码**:
```python
prompt += "\n[GUIDANCE] You must output valid JSON with exactly these fields: 'analysis', 'plan', 'commands'."
```

**为什么致命**:
- Terminus2 框架需要 `task_complete: true` 才能结束任务
- 这个 hook 告诉 agent "只能用这 3 个字段"，**直接抹掉了 agent 结束任务的能力**
- 导致所有任务在完成后仍然无限循环，直到超时

**影响范围**: 所有在第 3 轮后仍在运行的任务（几乎全部）

**修复建议**: 要么删除此 hook，要么改为 `'analysis', 'plan', 'commands', 'task_complete'`

---

### P0-2: `before_llm_call_..._d2d59c8d.py` — 用策略文本替换任务描述

**类型**: `before_llm_call`

**问题代码**:
```python
orig = context.original_instruction.strip()[:150]
prompt += f"Task: {orig}\n"
```

**为什么致命**:
- `context.original_instruction` 以 `[LEARNED STRATEGIES]` 开头（策略库全文）
- 截取前 150 字后变成 `Task: [LEARNED STRATEGIES] When: When creating a tar.gz archive of /home/user/data/...`
- **每一轮 agent 看到的"任务"都变成了 tar 打包而非真实任务**
- 这就是你看到的 `Task: [LEARNED STRATEGIES] When: When creating a tar.gz archive...` 的来源！

**影响范围**: 所有在第 5 轮后仍在运行的任务

**修复建议**: 删除此 hook，或修改为不截取 `original_instruction`（它包含策略前缀）

---

### P0-3: `after_round_..._3151307f.py` — 首次完成强制续轮

**类型**: `after_round`

**问题代码**:
```python
if kv['verify_count'] == 1:
    return {
        'request_new_turn': True,
        'next_prompt': '[VERIFICATION REQUIRED] You marked the task complete...'
    }
```

**为什么致命**:
- Agent 首次标记 `task_complete` 时，被强制要求"再验证一次"
- 与 P0-1 叠加后形成死锁：agent 无法发出 `task_complete` → 永远不触发二次确认逻辑 → 永远不结束

**影响范围**: 所有能到达完成阶段的任务

---

### P0-4: `after_round_..._b1215fca.py` — 另一个首次完成拦截器

**类型**: `after_round`

**问题代码**:
```python
if count == 0:
    return {
        "request_new_turn": True,
        "next_prompt": "[VERIFY BEFORE COMPLETE] You marked task_complete..."
    }
```

**为什么致命**: 与 P0-3 功能重复但来自不同 hook，两个 hook **同时拦截首次完成**，加重了死循环风险

---

### P0-5: `after_round_..._1fab62c7.py` — 第三个首次完成拦截器

**类型**: `after_round`

**问题代码**:
```python
if is_task_complete:
    if context.kv['complete_count'] == 1:
        return {
            'request_new_turn': True,
            'next_prompt': '[TASK COMPLETION VERIFICATION]...'
        }
```

**为什么致命**: **第三个**做同样事情的 hook。三个 hook 各自维护独立的完成计数器，互不感知，形成三重拦截

---

### P0-6: `before_execute_..._022ffa35.py` — 强制包裹 `su - user -c`

**类型**: `before_execute`

**问题代码**:
```python
new_keystrokes = 'su - user -c "' + cmd.keystrokes + '"'
```

**为什么致命**:
- 环境中**没有 `user` 用户**，直接导致命令失败
- 引号嵌套问题：如果原命令包含双引号，包裹后语法错误
- 在多个任务的 trial.log 中都能看到 `user user does not exist` 错误

---

## 四、严重问题 Hook (P1)

### P1-1: `before_llm_call_..._b871ce71.py` — 破坏性 prompt 改写

**类型**: `before_llm_call`

**问题**:
```python
prompt = prompt.replace('analysis:', '### ANALYSIS: ').replace('plan:', '### PLAN: ')
prompt += '\n\n[CRITICAL CONSTRAINTS]\n1. OUTPUT VALID JSON ONLY...\n[PREVIOUS APPROACH FAILED]...'
```

- 用 `replace` 破坏性修改 prompt 中所有 `analysis:` / `plan:` 字符串
- 谎称"之前的方法失败了"(`[PREVIOUS APPROACH FAILED]`)，即使之前没有失败
- 强制要求 `Use printf instead of heredoc`，与某些任务需求冲突

---

### P1-2: `before_llm_call_..._1506fcd9.py` — 任务描述被策略替换（[FOCUS]版）

**类型**: `before_llm_call`

**问题**:
```python
prompt += '\n[FOCUS] Keep working toward task completion. Original task: ' + context.original_instruction[:150]
```

与 P0-2 相同问题：`original_instruction[:150]` 是策略文本而非真实任务

---

### P1-3: `after_round_..._c4d0b384.py` — 过于激进的错误检测

**类型**: `after_round`

**问题**:
```python
if 'syntax' in terminal_output.lower() or 'error' in terminal_output.lower() or 'command' in terminal_output.lower():
    context.kv['parse_errors'] += 1
```

- 任何包含 "error"、"syntax"、"command" 的终端输出都会触发
- 正常输出如 `No syntax errors found` 也会误触发
- 触发后注入 `[PARSING ISSUES]` + 截断的策略文本当任务描述
- 附加 `request_new_turn: True` 强制续轮

---

### P1-4: `after_round_..._77441313.py` — [LOOP DETECTED] 带 request_new_turn

**类型**: `after_round`

**问题**: 检测到"循环"后同时发送 `request_new_turn: True` + `[LOOP DETECTED]`。
讽刺的是：**hook 自身导致了循环，然后又检测到循环并强制续轮，加重了循环**。

---

### P1-5: `after_round_..._5943530c.py` — 验证命令过度敏感

**类型**: `after_round`

**问题**:
```python
verify_cmds = ['cat', 'ls', 'head', 'tail', 'grep']
is_verifying = any(cmd in context.last_commands for cmd in verify_cmds)
```

- 子串匹配：`cat` 会匹配到 `concatenate`、`catalog` 等
- 3 次验证就触发 `[LOOP DETECTED]` + `request_new_turn`
- 正常工作流中 `cat` 和 `ls` 是常用命令，不应被视为"循环"

---

### P1-6: `before_execute_..._30f1f9d1.py` — 硬编码 `/home/user` 路径改写

**类型**: `before_execute`

**问题**:
```python
if '/' in stripped and not stripped.startswith('/'):
    new_line = '/home/user' + stripped
```

- 把相对路径强制改为 `/home/user/...`
- 大多数 terminal-bench 任务工作在 `/app` 目录，不是 `/home/user`
- 直接导致文件操作到错误路径

---

### P1-7: `after_execute_..._082538f5.py` — 硬编码任务编号

**类型**: `after_execute`

**问题**: 注释中写 `Task 944`，逻辑针对特定任务设计。在其他任务上可能产生无意义的截断或误判。

---

### P1-8: `before_llm_call_..._e1888498.py` — dpkg/checksum 特化逻辑

**类型**: `before_llm_call`

**问题**: 检测到 `dpkg` 或 `checksum` 关键词时注入特定格式说明，对不相关任务是纯噪声。

---

## 五、中等问题 Hook (P2)

### P2-1 ~ P2-3: 三个 `after_round` hook 同时做"循环检测"

文件: `c4d0b384`, `b1215fca`, `5943530c`

**问题**: 三个 hook **各自独立**检测循环，各自发送 `[LOOP DETECTED]` 和 `request_new_turn`，互不感知。一次"循环"可能触发 3 次强制续轮。

---

### P2-4 ~ P2-6: 多个 hook 向终端输出注入文本

文件: `after_execute_..._321b92b4`, `after_execute_..._e7ace1fd`, `after_execute_..._e8ed3dd8`

**问题**: 在 agent 看到的"终端输出"后面拼接 `[WARN]`、`[GUIDANCE]`、`[VERIFY]` 文本。Agent 可能把这些当作真实终端输出来理解。

---

### P2-7: `before_execute_..._c597eb6c.py` — 错误的命令替换处理

**问题**: `replace('$(', '(')` 试图消除命令替换但实际上会破坏合法的 `$(...)` 子命令

---

### P2-8 ~ P2-10: `on_timeout` hook 过度简化

文件: `a92a297b` (反对所有 printf)、`274f5346` (通用警告)、`be501739` (通用警告)

**问题**: `a92a297b` 看到任何 `printf` 超时就建议换用 `sed -i`，这是错误的泛化

---

## 六、低问题 Hook (P3)

| 文件 | 问题 |
|------|------|
| `after_execute_..._4d7741ab` | `recent = recent[-5:]` 不写回 kv，裁剪无效 |
| `after_execute_..._d5b1ecce` | 输出截断逻辑合理但阈值可调 |
| `after_execute_..._a23899ac` | 循环检测条件可能永不触发 |
| `before_execute_..._1c7ba974` | 静默丢弃 ls/find 命令可能让 agent 困惑 |
| `before_execute_..._77619f48` | 丢弃"校验类"后续命令，可能误删合法命令 |
| `before_execute_..._de3fac5a` | `&&`/`||` 警告逻辑无害但无用 |
| `before_execute_..._a846437a` | 硬编码 `/home/user/audit` 路径 |
| `before_execute_..._91619836` | 对 heredoc 注入注释行可能破坏命令结构 |
| `after_round_..._8d872b17` | `timeout`/`log_level` 关键词匹配过于特化 |
| `after_round_..._29d55b1a` | `[REFOCUS]` + 截断任务描述，与 P0-2 类似但影响较小 |
| `before_llm_call_..._2e0d547d` | `completion_attempts` 在文件内可能永不增加 |
| `before_llm_call_..._3ddce9d2` | 计数器依赖外部维护，可能永不触发 |
| `on_timeout_..._0672a5c8` | 警告写入 kv 但不返回给 agent，无效 |

---

## 七、根因总结

### 问题 1：`original_instruction` 被策略库前缀污染

**影响**: P0-2, P1-2, P1-3, P2-3 等所有引用 `context.original_instruction[:N]` 的 hook

**原因**: 策略库内容（`[LEARNED STRATEGIES]` + 32 条策略）被拼接在真实任务描述之前注入到 prompt。当 hook 截取 `original_instruction[:150]` 时，截到的是策略文本而非任务描述。

**修复方向**: 要么在注入策略库时不修改 `original_instruction`，要么 hook 中不引用 `original_instruction`。

### 问题 2：`[GUIDANCE]` 遗漏 `task_complete` 字段

**影响**: P0-1，导致所有超过 3 轮的任务无法正常结束

**修复方向**: 要么删除该 hook，要么补全字段列表。

### 问题 3：多个 after_round hook 竞争 `request_new_turn`

**影响**: 8 个 after_round hook 中至少 6 个会在不同条件下触发 `request_new_turn: True`，互不感知，叠加后强制续轮概率极高。

**修复方向**: 最多保留 1 个 after_round hook 负责续轮控制。

### 问题 4：硬编码路径和任务特化逻辑

**影响**: `/home/user`、`su - user -c`、`tar -tzf`、`dpkg`、`Task 944` 等在非目标任务上产生干扰。

**修复方向**: 删除所有硬编码路径和任务特化 hook，或在注入前检查任务类型。

---

## 八、建议删除/保留

### 建议立即删除（P0 + 部分 P1）
1. `before_llm_call_..._0bbc3050.py` — 删除 task_complete 字段
2. `before_llm_call_..._d2d59c8d.py` — 策略替换任务描述
3. `before_llm_call_..._b871ce71.py` — 破坏性 prompt 改写
4. `before_llm_call_..._1506fcd9.py` — [FOCUS] 策略替换
5. `before_llm_call_..._e1888498.py` — dpkg/checksum 特化
6. `before_execute_..._022ffa35.py` — 强制 su - user -c
7. `before_execute_..._30f1f9d1.py` — 硬编码 /home/user 路径
8. `after_round_..._3151307f.py` — 首次完成拦截
9. `after_round_..._b1215fca.py` — 首次完成拦截（重复）
10. `after_round_..._1fab62c7.py` — 首次完成拦截（重复）
11. `after_round_..._c4d0b384.py` — 过度激进错误检测
12. `after_round_..._5943530c.py` — 验证过度敏感
13. `after_round_..._77441313.py` — 循环检测 + 强制续轮
14. `after_execute_..._082538f5.py` — 硬编码 Task 944

### 建议保留但修改
- `after_round_..._8d872b17.py` — 去掉特化关键词
- `after_round_..._29d55b1a.py` — 不截取 original_instruction
- `after_execute_..._d5b1ecce.py` — 输出截断逻辑合理
- `on_timeout` 类 — 保留通用警告，删除 printf 特化

### 可保留无害 Hook
- `before_execute_..._de3fac5a.py` — 空命令过滤
- `after_execute_..._4d7741ab.py` — 输出截断（修 bug）
- `on_timeout_..._e2ba4037.py` — 通用超时提示
