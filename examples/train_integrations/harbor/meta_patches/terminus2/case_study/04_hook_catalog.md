# Meta Hook 完整目录：37 个 Hook 逐一介绍

> 共 37 个 hook，按触发点分为 5 类：`before_llm_call`（8 个）、`before_execute`（8 个）、`after_round`（8 个）、`after_execute`（8 个）、`on_timeout`（5 个）

---

## 一、before_llm_call — LLM 调用前修改提示词（8 个）

这类 hook 在每次调用 LLM 之前执行，可以修改发给模型的 prompt。

### 1. `before_llm_call_..._e8f92ae8.py`

**作用**：检测 agent 是否在重复执行 `cat`（针对 `/home/user/` 路径）或大量 `echo`/`printf` 命令。如果 `cat` 命中 ≥3 次，注入 `[WARN]` 提示减少重复验证；如果 `echo`/`printf` ≥5 次，建议用单条 `printf` 替代多条。

**注入标签**：`[WARN]`

**问题**：
- 硬编码了 `/home/user/` 路径，大部分 terminal-bench 任务工作在 `/app`，匹配不到
- `echo`/`printf` ≥5 次的阈值对某些需要逐行写文件的任务过于敏感

---

### 2. `before_llm_call_..._0bbc3050.py`

**作用**：在第 3 次 LLM 调用后，注入 `[GUIDANCE]` 强制要求 agent 输出合法 JSON，字段限定为 `analysis`、`plan`、`commands`。同时检测 prompt 中是否出现 "Are you sure" 或 "verify"，若验证次数 >2 则提示信任自己的工作。

**注入标签**：`[GUIDANCE]`、`[WARNING]`

**⚠️ 致命问题**：
- **`[GUIDANCE]` 中列出的字段缺少 `task_complete`**！Terminus2 框架需要 `"task_complete": true` 才能结束任务，但 hook 告诉 agent "只能用 analysis/plan/commands 三个字段"，导致 agent 永远无法正常标记完成
- 这是导致 constraints-scheduling 等任务陷入 168 轮死循环的**根本原因之一**

---

### 3. `before_llm_call_..._3ddce9d2.py`

**作用**：追踪 `verification_attempts` 和 `task_complete_attempts` 两个计数器。当验证尝试 ≥2 次或完成尝试 ≥2 次时，注入 `[DEBUG]` 提示 agent 确认是否满足所有要求。

**注入标签**：`[DEBUG]`

**问题**：
- 这两个计数器在**本 hook 内从未递增**，依赖其他 hook 写入 kv，但tm键名不一定匹配，很可能永远不触发

---

### 4. `before_llm_call_..._b871ce71.py`

**作用**：每 5 轮 LLM 调用后，用 `str.replace()` 将 prompt 中的 `analysis:`、`plan:`、`commands:` 替换为 `### ANALYSIS:` 等 Markdown 格式，然后追加 `[CRITICAL CONSTRAINTS]` 和 `[PREVIOUS APPROACH FAILED]` 大段约束文本。

**注入标签**：`[CRITICAL CONSTRAINTS]`、`[PREVIOUS APPROACH FAILED]`

**⚠️ 严重问题**：
- `str.replace()` 是全文替换，会**破坏**所有包含 `analysis:`、`plan:`、`commands:` 的 prompt 文本（包括终端输出等）
- `[PREVIOUS APPROACH FAILED]` 是谎言——之前的方法可能完全没有失败
- 强制要求 "Use printf instead of heredoc"，对需要 heredoc 的任务有害
- 重置计数器 `verbose_call_count = 0`，使其每 5 轮触发一次，反复注入

---

### 5. `before_llm_call_..._d2d59c8d.py`

**作用**：追踪 LLM 调用次数，在第 5 次调用后注入 `[CONTEXT CONSTRAINT]`，提示 agent 接近上下文限制，并附上任务描述的前 150 字。

**注入标签**：`[CONTEXT CONSTRAINT]`

**⚠️ 致命问题**：
- `context.original_instruction[:150]` 截取的是**策略库文本**而非真实任务描述！因为策略库内容被拼接在 original_instruction 的开头
- 导致 agent 每轮看到 `Task: [LEARNED STRATEGIES] When: When creating a tar.gz archive of /home/user/data/...` 而非真实任务
- **这就是你看到 `Task: [LEARNED STRATEGIES] When: When creating a tar.gz archive...` 的直接原因**

---

### 6. `before_llm_call_..._e1888498.py`

**作用**：仅在第 0 轮（首轮）检查任务描述中是否包含 `dpkg` 或 `checksum` 关键词。若包含 `dpkg`，提示使用重定向 `>` 将输出写入文件；若包含 `checksum`，提示使用"两个空格"分隔哈希和文件名。

**注入标签**：`[IMPORTANT]`

**问题**：
- 纯任务特化逻辑，仅对 dpkg/checksum 类任务有用，对其他 87 个任务完全无效
- 不会造成破坏，但属于冗余

---

### 7. `before_llm_call_..._1506fcd9.py`（s10 版本）

**作用**：追踪 prompt 中的 `cat`/`ls` 命令模式，检测重复验证。若检测到重复，注入 `[WARNING]`。在第 10 次调用后注入 `[FOCUS]`，附上任务描述前 150 字。

**注入标签**：`[WARNING]`、`[FOCUS]`

**问题**：
- 重复检测逻辑有 bug：`all(s in [last_3[0], last_3[1], last_3[2]] for s in [last_3[-1]])` 基本恒真
- `[FOCUS]` 使用 `context.original_instruction[:150]` → 同 P0-2，截取到的是策略文本

---

### 8. `before_llm_call_..._2e0d547d.py`（s14 版本）

**作用**：追踪完成尝试次数和验证次数。验证 ≥3 次时提示停止验证；完成尝试 ≥2 次时注入 `[RESTART]`，附上任务描述的前 500 字，要求重新阅读。

**注入标签**：`[WARNING]`、`[RESTART]`

**问题**：
- `completion_attempts` 在本文件内从未递增，`[RESTART]` 分支可能永远不触发
- `[RESTART]` 使用 `context.original_instruction[:500]`，同样会截取到策略文本

---

## 二、before_execute — 命令执行前修改命令列表（8 个）

这类 hook 在 agent 输出的命令被执行前运行，可以修改、过滤或添加命令。

### 9. `before_execute_..._1c7ba974.py`

**作用**：统计并过滤"肤浅检查"命令。如果 `ls`/`find`/`dir` 命令超过 3 次，**静默丢弃**后续同类命令；同时统计 `cat`/`head`/`grep` 的验证次数（但不丢弃）。

**问题**：
- 静默丢弃命令会让 agent 困惑——它发出了命令但没有收到输出
- 计数器 `superficial_check_count` 跨任务/轮次累加，不重置

---

### 10. `before_execute_..._022ffa35.py`

**作用**：在首次遇到文件操作命令（touch/echo/cat/printf/sed 等）时，自动将命令包裹为 `su - user -c "原命令"`。

**⚠️ 致命问题**：
- terminal-bench 环境中**没有 `user` 用户**，`su - user -c` 直接导致命令失败
- 双引号嵌套：如果原命令包含双引号，包裹后语法错误
- 在多个任务轨迹中都看到 `user user does not exist` 错误

---

### 11. `before_execute_..._30f1f9d1.py`

**作用**：尝试将相对路径转换为绝对路径，方法是在路径前拼接 `/home/user`。

**⚠️ 严重问题**：
- 硬编码 `/home/user` 前缀，但大部分任务工作在 `/app` 目录
- 启发式判断非常粗糙（`stripped.startswith('/') == False and not stripped[0].isalpha()`），会误伤非路径命令
- 例如 `./configure` 会被改成错误的路径

---

### 12. `before_execute_..._77619f48.py`

**作用**：限制连续验证命令数量——连续 2+ 个 `cat`/`ls`/`wc`/`grep`/`find` 命令时丢弃后续的。同时检测含 `EOF` 的 heredoc 命令，若 `complex_cmd_count` ≥2 则丢弃。

**问题**：
- 丢弃 heredoc 命令的逻辑有 bug：`complex_cmd_count` 初始为 0，`setdefault` 后立即 `+= 1`，但 `continue` 跳过了赋值
- 对正常工作流中"先 cat 查看文件再 grep 搜索"的模式会误伤

---

### 13. `before_execute_..._c597eb6c.py`

**作用**：过滤空命令；尝试处理含 `$(` 或反引号的命令替换（用 `replace('$(', '(')` 消除）；将命令 duration 上限设为 30 秒；命令数量 >5 时标记停止循环。

**问题**：
- `replace('$(', '(')` 不能正确消除命令替换，反而会**破坏**合法的 `$(...)` 子命令
- `stop_loop` 标记被设置但没有任何地方读取它

---

### 14. `before_execute_..._de3fac5a.py`

**作用**：过滤空命令；统计含 `&&` 或 `||` 的复合命令数量。

**问题**：基本无害，只是 `complex_cmd_warning` 计数器没有被任何其他地方消费

---

### 15. `before_execute_..._a846437a.py`（s10 版本）

**作用**：去重 `cat`/`ls`/`head` 命令；在 `find` 命令前自动插入 `mkdir -p /home/user/audit &&`；将 duration 上限设为 30 秒。

**问题**：
- 硬编码 `/home/user/audit` 路径，与大多数任务的工作目录不符
- 在 `find` 命令前插入 `mkdir` 可能破坏原命令意图

---

### 16. `before_execute_..._91619836.py`（s14 版本）

**作用**：过滤空命令；丢弃以 `>` 结尾的畸形命令；**丢弃超过 3 行的命令**；对包含 `EOF` 的 heredoc 命令在前面插入注释行 `# Use simple echo instead of heredoc`。

**问题**：
- **丢弃超过 3 行的命令**过于激进——很多合法的 Python 脚本或配置文件创建需要多行
- 在 heredoc 前插入注释行会破坏 heredoc 的语法结构

---

## 三、after_round — 每轮结束后的控制逻辑（8 个）

这类 hook 在每轮交互结束后执行，可以注入下一轮提示、控制是否续轮（`request_new_turn`）、检测任务完成状态。

### 17. `after_round_..._1fab62c7.py`

**作用**：多功能 hook。检测脚本缺少 shebang 行（基于 kv 中的标记）→ 注入 `[CRITICAL]` 提示；检测 heredoc 结构问题 → 注入 `[COMMAND STRUCTURE ISSUE]`；解析错误过多 → 注入 `[PARSE ERROR FREQUENCY HIGH]`；命令执行超过 15 条 → 注入 `[CONTEXT WARNING]`；**首次标记 task_complete 时，强制 `request_new_turn: True` 并要求额外验证**。

**注入标签**：`[CRITICAL]`、`[COMMAND STRUCTURE ISSUE]`、`[PARSE ERROR FREQUENCY HIGH]`、`[CONTEXT WARNING]`、`[TASK COMPLETION VERIFICATION]`

**⚠️ 致命问题**：
- 首次完成时 `request_new_turn: True` 强制续轮，与 #21、#23 形成**三重拦截**
- 依赖其他 hook 写入 kv（如 `shebang_missing`、`heredoc_issues`、`parse_issues`），如果上游 hook 不写入则永远不触发

---

### 18. `after_round_..._8d872b17.py`

**作用**：当 agent 标记 task_complete 且任务描述中包含 `timeout` 或 `log_level` 关键词时，注入 `[VERIFY]` 确认提示。

**注入标签**：`[VERIFY]`

**问题**：
- 纯任务特化逻辑，仅对少数包含 timeout/log_level 的任务有效
- 基本无害但冗余

---

### 19. `after_round_..._c4d0b384.py`

**作用**：最复杂的 after_round hook。包含 5 个独立检测模块：(1) 验证循环检测 → `[LOOP DETECTED]` + `request_new_turn`；(2) 卡住状态检测 → `[STUCK STATE]` + `request_new_turn`；(3) 接近 episode 上限提醒 → `[SPEED WARNING]`；(4) 首次 task_complete → 要求再次验证；(5) 命令解析错误检测 → `[PARSING ISSUES]`。

**注入标签**：`[LOOP DETECTED]`、`[STUCK STATE]`、`[SPEED WARNING]`、`[TASK COMPLETE]`、`[PARSING ISSUES]`

**⚠️ 严重问题**：
- 错误检测过于宽泛：`'error' in terminal_output.lower()` 会匹配到 `No errors found` 等正常输出
- 多处使用 `context.original_instruction[:150]` → 截取到策略文本
- 首次完成时再次要求验证，与 #17、#21、#23 叠加
- `near_limit_warned = True` 写在 `return` 语句之后，永远执行不到

---

### 20. `after_round_..._29d55b1a.py`

**作用**：当 agent 多次标记 task_complete 时，注入 `[REFOCUS]` 并附上任务描述的前 200 字。

**注入标签**：`[REFOCUS]`

**问题**：
- 使用 `context.episode.original_instruction[:200]`，这里的 `context.episode` 可能不是正确的属性访问方式（应该是 `context.original_instruction`），可能抛出 AttributeError
- 还检测了 `timeout (TO)` 和 `log_level (LOG_LEVEL)` 关键词，属于任务特化

---

### 21. `after_round_..._3151307f.py`

**作用**：**首次**标记 task_complete 时，发送 `request_new_turn: True` 加 `[VERIFICATION REQUIRED]` 提示，要求 agent 再验证一次。**第二次**标记完成时，返回 `request_new_turn: False` 允许真正完成。非完成状态时重置计数器。

**注入标签**：`[VERIFICATION REQUIRED]`

**⚠️ 致命问题**：
- 这个 hook 的"两步确认"设计本身看似合理，但**与 P0-1（`[GUIDANCE]` 删除 task_complete 字段）叠加后形成死锁**：agent 永远无法发出 `task_complete: true`，所以 `is_task_complete` 永远为 false，这个 hook 的 verify_count 就永远停在 0，永远不会走到"第二次允许完成"的分支

---

### 22. `after_round_..._77441313.py`

**作用**：追踪完成尝试次数和验证次数。当验证尝试 ≥2 且再次标记完成时，发送 `request_new_turn: True` 加 `[LOOP DETECTED]`，提示 agent 停止重复验证。若使用了过多复杂命令，注入 `[WARNING]`。

**注入标签**：`[LOOP DETECTED]`、`[WARNING]`

**问题**：
- 本意是打破循环，但 `request_new_turn: True` 实际上**加重了循环**
- 与 #17、#19、#21、#23 的完成检测逻辑重复冲突

---

### 23. `after_round_..._b1215fca.py`

**作用**：三合一 hook：(1) 验证次数 >3 时注入 `[GUIDANCE]`；(2) **首次** task_complete 时 `request_new_turn: True` + `[VERIFY BEFORE COMPLETE]`，第二次允许完成；(3) 检测最近 4 条命令中是否有重复模式 → `[LOOP DETECTED]`。

**注入标签**：`[GUIDANCE]`、`[VERIFY BEFORE COMPLETE]`、`[LOOP DETECTED]`

**⚠️ 致命问题**：
- **第四个**做首次完成拦截的 hook（与 #17、#19、#21 一起形成四重拦截）
- 第二次完成时将 `complete_count` 重置为 0，意味着下下次完成又会被拦截

---

### 24. `after_round_..._5943530c.py`（s14 版本）

**作用**：检测验证命令（`cat`/`ls`/`head`/`tail`/`grep`）的使用次数。连续 3+ 次验证命令 → `request_new_turn: True` + `[LOOP DETECTED]`。同时检测命令中 `>` 出现超过 2 次 → `[COMMAND ERROR]` + `request_new_turn`。

**注入标签**：`[LOOP DETECTED]`、`[COMMAND ERROR]`

**问题**：
- 验证检测用 `any(cmd in context.last_commands for cmd in verify_cmds)` — 子串匹配，`cat` 会匹配到包含 "cat" 的任何字符串（如 `concatenate`）
- `verification_count` 作为局部变量递增但**没写回 kv**，下次调用时又从 kv 读旧值
- `context.original_instruction[:150]` → 策略文本问题

---

## 四、after_execute — 命令执行后处理终端输出（8 个）

这类 hook 在命令执行完毕后运行，可以修改 agent 看到的终端输出。

### 25. `after_execute_..._4d7741ab.py`

**作用**：检测终端输出中的文件创建操作，标记缺少 shebang 的脚本（仅当同时包含 `.sh`、`echo` 和 `sudo ufw`）；检测 heredoc 中的畸形重定向；追踪解析错误和 rsync/scp 操作。

**问题**：
- shebang 检测条件过于苛刻（必须同时有 `.sh`、`echo`、`sudo ufw`），几乎不会触发
- `recent = recent[-5:]` 不写回 kv，截断无效

---

### 26. `after_execute_..._d909fbbb.py`

**作用**：追踪最近命令；检测终端"卡住"状态（输出中包含 `>` 或以 `...` 结尾 或过短）；追踪文件创建操作；检测错误输出；对超过 3000 字的输出进行截断并标记 `[OUTPUT TRUNCATED]`。

**问题**：
- `>` 在正常 shell 输出中非常常见（如重定向提示），误判率极高
- `recent = recent[-5:]` 同样不写回 kv

---

### 27. `after_execute_..._082538f5.py`

**作用**：注释中标记为 "Task 944" 专用。截断超过 4000 字的输出；检测多行输出标记 `multi_line_output`；追踪 `Permission denied` 和 `command not found` 错误计数。

**问题**：
- 硬编码 Task 944 特化逻辑
- `cmd_parse_flags` 不断追加 `"multi_line_output,"` 字符串，永远增长

---

### 28. `after_execute_..._321b92b4.py`

**作用**：检测终端输出中的 `heredoc`/`EOF` 关键词，累计 heredoc 尝试次数。超过 2 次后在终端输出末尾追加 `[WARN] Heredoc commands may fail in this environment. Consider using simple echo/printf instead.`

**注入标签**：`[WARN]`（追加到终端输出）

**问题**：
- `EOF` 在很多非 heredoc 的上下文中也会出现（如 "end of file" 错误信息）
- 向终端输出注入文本会让 agent 误以为这是真实的终端反馈

---

### 29. `after_execute_..._a23899ac.py`

**作用**：追踪 log 文件的写入历史。检测连续 3 次相同的 `echo`/`printf` 命令 → 在输出末尾追加 `[LOOP DETECTED]`。

**注入标签**：`[LOOP DETECTED]`（追加到终端输出）

**问题**：
- `'write' in context.last_commands` — `write` 作为子串匹配可能永远不满足
- `history = history[-20:]` 不写回 kv

---

### 30. `after_execute_..._d5b1ecce.py`

**作用**：截断超过 3000 字的终端输出（保留首尾各 1000 行）；追踪验证类命令（`cat`/`ls`/`head`/`tail`/`wc`/`grep`）的出现次数；次数 >3 时在 kv 中记录 `[WARNING]`。

**问题**：
- 截断逻辑合理但按行数（1000）而非字符数，可能导致保留内容仍然很长
- `loop_warning` 写入 kv 但没有任何地方消费它

---

### 31. `after_execute_..._e7ace1fd.py`

**作用**：检测终端输出中的验证命令模式。如果同类验证命令连续出现 3 次，在输出末尾追加 `[GUIDANCE]`，带 ANSI 颜色代码（黄色高亮）。

**注入标签**：`[GUIDANCE]`（追加到终端输出，带 ANSI 转义）

**问题**：
- ANSI 颜色转义码 `\x1b[33m...\x1b[0m` 可能干扰 LLM 的文本解析
- 检测逻辑 `detected in recent` 比较的是列表对象而非内容，可能永远不匹配

---

### 32. `after_execute_..._e8ed3dd8.py`

**作用**：检测 cron 相关操作。如果终端输出包含 `su` 和 cron 相关内容，且未包含 `crontab -l`，在输出末尾追加 `[VERIFY]` 提示运行 `crontab -l` 验证。

**注入标签**：`[VERIFY]`（追加到终端输出）

**问题**：
- 纯 cron 任务特化逻辑，对其他任务无效
- 向终端输出注入文本可能造成混淆

---

## 五、on_timeout — 命令超时后处理（5 个）

这类 hook 在命令执行超时后运行，可以在终端输出后追加提示信息。

### 33. `on_timeout_..._a92a297b.py`

**作用**：如果超时的命令中包含 `printf`，在输出末尾追加 `[TIMEOUT] WARNING: Using printf for file modifications may cause silent failures. Consider using sed -i for in-place edits.`

**注入标签**：`[TIMEOUT]`

**问题**：
- 将所有 `printf` 超时归咎于 printf 本身并建议换用 `sed -i`，这是错误的泛化
- `printf` 超时更可能是因为内容过长或管道阻塞，与 printf 工具本身无关

---

### 34. `on_timeout_..._0672a5c8.py`

**作用**：统计超时次数。如果超时命令长度 >100 字符，在 kv 中记录 `[WARNING]` 建议拆分命令。

**问题**：
- 警告写入 kv 但**不返回给 agent**（`return terminal_output` 没有追加警告），所以 agent 永远看不到这个警告，hook 实际无效

---

### 35. `on_timeout_..._274f5346.py`

**作用**：统计超时次数。第 2 次超时后在输出末尾追加 `[TIMEOUT WARNING] Multiple timeouts detected. Simplify commands and avoid long-running operations.`

**注入标签**：`[TIMEOUT WARNING]`

**问题**：基本无害，仅在多次超时后提示

---

### 36. `on_timeout_..._e2ba4037.py`

**作用**：无条件在超时输出末尾追加 `[TIMEOUT] Command timed out. The command may be too complex or missing proper syntax. Try breaking it into smaller steps.`

**注入标签**：`[TIMEOUT]`

**问题**：基本无害，是通用的超时提示

---

### 37. `on_timeout_..._be501739.py`（s14 版本）

**作用**：无条件在超时输出末尾追加 `[TIMEOUT] This command was too long or complex. Break it into smaller steps.`

**注入标签**：`[TIMEOUT]`

**问题**：与 #36 功能完全重复

---

## 六、统计总结

### 按触发点统计

| 触发点 | 数量 | 主要作用 |
|--------|------|---------|
| `before_llm_call` | 8 | 修改 LLM 输入提示词 |
| `before_execute` | 8 | 过滤/修改待执行命令 |
| `after_round` | 8 | 控制是否续轮、注入下轮提示 |
| `after_execute` | 8 | 处理/截断终端输出 |
| `on_timeout` | 5 | 超时后追加提示 |

### 按问题严重度统计

| 级别 | 数量 | Hook 编号 |
|------|------|----------|
| **致命 (P0)** | 6 | #2, #5, #10, #17, #21, #23 |
| **严重 (P1)** | 8 | #4, #7, #11, #19, #22, #24, #27, #6(轻) |
| **中等 (P2)** | 10 | #1, #9, #12, #13, #15, #16, #28, #29, #31, #33 |
| **低/无害 (P3)** | 13 | #3, #8, #14, #18, #20, #25, #26, #30, #32, #34, #35, #36, #37 |

### `request_new_turn: True` 的 Hook

以下 hook 会强制触发新一轮 LLM 调用，是导致 episode 膨胀和超时的主因：

| Hook | 触发条件 |
|------|---------|
| #17 `1fab62c7` | 首次 task_complete |
| #19 `c4d0b384` | 验证循环 ≥3 / 卡住 ≥3 / 首次 task_complete |
| #21 `3151307f` | 首次 task_complete |
| #22 `77441313` | verify_attempts ≥2 + 再次 complete |
| #23 `b1215fca` | 首次 task_complete |
| #24 `5943530c` | 连续验证 ≥3 / 复杂重定向 |

**6 个 hook 可以独立触发 `request_new_turn`**，在最坏情况下一轮结束后所有 6 个同时触发，造成灾难性的 episode 膨胀。
