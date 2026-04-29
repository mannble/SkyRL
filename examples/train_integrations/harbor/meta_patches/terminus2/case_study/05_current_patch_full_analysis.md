# 当前 Active Meta Patch 全量分析

分析对象：`/home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus2`

范围：
- `strategy_library.yaml`：32 条 active strategy。
- `hooks/*.py`：37 个 active hook。
- 不包括 `.backup/` 里的旧补丁，也不包括已有 `case_study/` 文档。

风险等级说明：
- P0：强烈建议删除或禁用。会直接破坏任务完成、命令执行或导致续轮死循环。
- P1：高风险。可能明显降低 terminal-bench 分数，需要重写后才能保留。
- P2：中风险。有一定作用，但触发条件粗糙或容易污染上下文。
- P3：低风险。主要是记录状态或轻量提醒，收益有限但一般不致命。

## 总体结论

这批补丁的主要问题不是“完全没有价值”，而是把从少数任务中学到的经验直接全局注入到了所有任务里。策略库里有很多任务特化路径，例如 `/home/user/data`、`pg-optimize`、`daily_summary_2024-04-12.txt`、`src_backup_2023.tar.gz`，这些经验在对应任务上可能有效，但在其他任务上会把模型拉向错误操作。

Hook 的风险更集中。最危险的是三类：

1. 强制模型输出字段只有 `analysis / plan / commands`，漏掉 `task_complete`。
2. `before_execute` 直接改写或丢弃命令，例如包成 `su - user -c`、硬拼 `/home/user`、丢弃多行命令。
3. 多个 `after_round` 同时在首次 `task_complete` 后发起 `request_new_turn`，会把“完成任务”变成“继续验证”，造成 episode 膨胀甚至死循环。

当前建议：不要整包启用。若要保留，应只留下少量通用策略，并把所有能改命令或强制续轮的 hook 放进更严格的 canary/arbiter 审核。

## Strategy Library 分析

### S01 `tar.gz /home/user/data` 根目录归档策略

作用：
- 针对 `/home/user/data` 下 `alpha.txt / beta.txt` 的打包任务，要求先 `cd /home/user/data`，让 archive root 直接包含文件名。
- 包含 `tar -tzf` 和 `stat` 验证。

可能收益：
- 对这个具体归档任务非常有帮助，能避免 `data/alpha.txt` 这种目录前缀错误。

风险：
- 强烈任务特化，路径、文件名、输出 archive 路径都硬编码。
- 全局注入时会误导非归档任务，尤其曾经导致 hook 截取 `context.original_instruction[:150]` 时把这条策略当成“原始任务”。

建议：
- P1。只应作为检索式策略，在任务描述明确出现 `/home/user/data` 和 archive root 要求时启用。

### S02 备份 archive 解压验证策略

作用：
- 创建 `/tmp/verify`，解压归档，使用 `diff -r` 对比原目录。
- 同时检查 `backup_log.txt` 和 symlink。

可能收益：
- 对备份/恢复类任务有价值，鼓励真实验证而不是只看 tar 列表。

风险：
- `/path/to/archive.tar.gz` 是占位符，模型可能原样照抄。
- 要求 `backup_log.txt` 和 `latest.tar.gz`，但并非所有 archive 任务都有这些产物。

建议：
- P2。可保留为“备份类任务泛化策略”，但必须通过任务检索过滤。

### S03 Python 优化 solver 运行验证策略

作用：
- 运行 solver 脚本，检查 `plan.json`、`run.log` 和最优 cost。

可能收益：
- 对优化/规划类任务有价值，提醒 agent 不要只写脚本不运行。

风险：
- 假设输出文件名固定为 `plan.json` 和 `run.log`。
- “optimal cost” 容易让非优化任务产生无关检查。

建议：
- P2。保留时应改成通用模板：运行生成脚本并检查任务要求的输出文件。

### S04 按行号修改配置文件策略

作用：
- 修改配置前备份。
- 用 `sed -i '7s/^/#/'` 这种行号方式改文件。
- 检查行号、规则计数，并要求创建 6 行 `remediation.log`。

可能收益：
- 对 firewall/config 精确行号任务有帮助。

风险：
- 行号 `7`、规则 `23`、`-A INPUT.*ACCEPT`、`remediation.log` 都高度特化。
- 对任意配置任务使用会直接改错行。

建议：
- P1。只适合保存在任务记忆库，不适合作为全局策略。

### S05 Makefile tab 缩进策略

作用：
- 强调 Makefile recipe 行必须用 tab。
- 推荐 `printf` 或 heredoc 创建 Makefile，并用 `cat -A` 验证 `^I`。

可能收益：
- 通用性较好，Makefile 任务常见失败点就是空格/tab。

风险：
- 示例 `sed -i 's/^\(\t\)/\t/'` 没有实际修复“空格变 tab”的能力。
- 同时推荐 heredoc 和全局“避免 heredoc”策略冲突。

建议：
- P2。保留，但删掉无效 sed 示例，明确只在 Makefile 任务启用。

### S06 忽略 `timeout/TO` 和 `log_level/LOG_LEVEL` 提示策略

作用：
- 当验证系统询问 timeout/log_level 时，提醒这些可能不是原任务要求。
- 鼓励只验证原始任务产物。

可能收益：
- 对曾经出现误导性 completion prompt 的任务有帮助。

风险：
- 如果真实任务确实要求 `timeout` 或 `log_level`，这条会让 agent 忽略关键要求。

建议：
- P2。应改成“若原始任务未提到这些字段则忽略”，不要无条件忽略。

### S07 dpkg 输出写入 log 策略

作用：
- 要求第一条生成内容的命令就带 `>` 重定向。
- 给出 `dpkg -l bash | grep ... > /home/user/utilities/bash-package.log` 示例。

可能收益：
- 对 dpkg 日志任务很有效，避免只打印不写文件。

风险：
- 完全特化到 `bash-package.log` 和 `/home/user/utilities`。
- 全局注入会导致模型在非 dpkg 任务中倾向使用 dpkg 或错误路径。

建议：
- P1。只允许关键词检索命中 `dpkg` 任务时启用。

### S08 Docker 容器内数据库查询策略

作用：
- 检查 docker 是否可用。
- 使用 `docker exec <container> psql ...` 在容器内执行数据库查询。
- 将 query plan 重定向到文件。

可能收益：
- 对数据库在容器内的任务很有价值。

风险：
- 假设容器名、用户、数据库、查询工具都是 PostgreSQL。
- terminal-bench 里未必允许 docker 或容器名不同。

建议：
- P2。适合保留为检索式策略，触发条件必须包含 Docker/container/database。

### S09 daily summary 精确 printf 策略

作用：
- 用固定 `printf` 创建 `/home/user/analysis/daily_summary_2024-04-12.txt`。
- 固定内容包括 `DATE`、`TASK_START`、`TASK_SUCCESS` 等。

可能收益：
- 对这个具体 log summary 任务有用。

风险：
- 日期、路径、字段值全部硬编码。
- 会在其他 log summary 任务中生成错误内容。

建议：
- P1。强任务特化，不适合全局注入。

### S10 rsync stats 文件数修补策略

作用：
- 如果 rsync log 显示 `Number of files: 5`，直接用 sed 改成 `4`。

可能收益：
- 可能绕过某个具体任务的统计格式问题。

风险：
- 这是最危险的策略之一：它鼓励篡改日志而不是修正备份命令。
- 若真实文件数确实是 5，会生成虚假结果。

建议：
- P0。建议删除。terminal-bench 目标是完成任务，不应学习“改日志骗验证”。

### S11 SSH keypair 权限策略

作用：
- 创建 `.ssh`，设置目录 700，生成 Ed25519 key，私钥 600。

可能收益：
- SSH key 任务常见且权限要求稳定，比较有泛化价值。

风险：
- 文件名 `id_microservice_ed25519` 和 `/home/user/.ssh` 特化。
- 某些任务需要不同 comment、passphrase 或 authorized_keys。

建议：
- P2。保留时应参数化文件名和目标用户。

### S12 command_parsing_failure / verification loops 汇总策略

作用：
- 避免 `su - user -c`。
- 简化命令，一次综合验证，减少连续 cat。

可能收益：
- 方向正确，针对之前的命令解析和过度验证问题。

风险：
- “不要连续 cat”如果过度执行，会阻止必要的逐文件检查。
- “Are you sure 多次出现时运行 full verification”可能与 after_round 强制验证叠加。

建议：
- P2。可保留，但要弱化成“避免无意义重复验证”。

### S13 DNS resolution audit log 策略

作用：
- 读取 `expected_hosts.txt`，用 `getent hosts` 比对，生成 `dns_resolution_audit.log`。

可能收益：
- 对 DNS audit 特定任务有用。

风险：
- 硬编码 `localhost / ip6-localhost`、路径和输出格式。
- 不适合其他网络/DNS 任务。

建议：
- P1。只保留在任务检索库。

### S14 shell script 特殊字符 printf 策略

作用：
- 使用 `printf '%s\n'` 和单引号避免 `%`、`$` 等特殊字符被 shell 展开。
- 建议立即 `cat -A` 验证。

可能收益：
- 通用价值较高，确实能减少复杂 quoting 错误。

风险：
- “用 echo 增量构建脚本”和其他策略“不要多次 echo/printf”冲突。
- “限制 10 turn”是硬性行为建议，可能让复杂任务过早完成。

建议：
- P2。保留核心 quoting 建议，删除 turn 数硬限制。

### S15 shell script 增量 echo 替代 heredoc 策略

作用：
- 用多条 echo 逐行创建 shell script。
- 避免 heredoc。

可能收益：
- 对短脚本可减少 heredoc 解析失败。

风险：
- 与 S16 “不要多条 echo/printf 构建同一文件”直接冲突。
- 对长脚本会膨胀命令数和上下文。

建议：
- P2/P1。保留为短脚本策略，但不能全局压制 heredoc。

### S16 多行文件用单个 printf 替代 heredoc/echo

作用：
- 推荐单条 `printf` 生成多行文件。
- 反对 heredoc，也反对多条 echo/printf。

可能收益：
- 对短文本文件有效，减少 command count。

风险：
- 对长文件、包含引号/反斜杠/百分号的内容，单个 printf 更容易出错。
- 与 S15 冲突。

建议：
- P2。需要条件化：短内容用 printf，长内容可用 heredoc 或脚本生成。

### S17 避免复杂 command chaining / `su - user -c`

作用：
- 建议用简单单行命令，避免复杂 `|`、`&&`、`||`。
- 提到如果用 `su - user -c` 要正确 quote。

可能收益：
- “避免 `su - user -c` 干扰”是正确教训。

风险：
- 又保留了 `su - user -c` 的可能性，会和有害 hook 形成呼应。
- 禁用 pipes 会伤害很多文本处理任务。

建议：
- P2。改成“不要自动包 `su - user -c`；需要用户切换时先确认用户存在”。

### S18 反复 verification prompt 处理策略

作用：
- 出现 3 次以上 verification prompt 时，只做一次综合验证，然后完成。

可能收益：
- 对 completion loop 有帮助。

风险：
- 如果前几次验证失败，这条可能鼓励过早提交。
- 与多个 after_round hook 叠加时会制造“验证一次后又被要求验证”的循环。

建议：
- P2。保留思想，但应让 runtime 限制 after_round 续轮数量。

### S19 archive + checksum 策略

作用：
- 创建 archive，排除 `model.bin`，生成 checksum manifest。

可能收益：
- 对某个 ML artifact 任务有用。

风险：
- 路径、文件名、archive 名全部硬编码。

建议：
- P1。任务特化，只做检索式策略。

### S20 15+ turns context limit 文件创建策略

作用：
- 接近上下文限制时合并文件创建和验证命令。
- 通过一次 `ls -la && cat` 完成检查。

可能收益：
- 方向正确，能减少上下文膨胀。

风险：
- 15 turns 阈值不一定可靠。
- “trust previous verification”可能导致遗漏。

建议：
- P2。适合作为软提示，不应由 hook 强制。

### S21 complex quoting 解析失败策略

作用：
- 和 S12 几乎重复：简化命令、避免 `su - user -c`、一次综合验证。

可能收益：
- 通用方向正确。

风险：
- 与 S12 重复，增加 prompt 噪声。

建议：
- P2。应与 S12 合并去重。

### S22 多文件格式任务一次性处理策略

作用：
- 创建 logs/quarantine/summary JSON，设置权限。

可能收益：
- 对具体 validation/quarantine/release-summary 任务有用。

风险：
- 强任务特化。全局注入可能让模型凭空创建这些文件。

建议：
- P1。只做任务检索策略。

### S23 SSH keys 多步骤分开执行策略

作用：
- 创建 `.ssh`，生成 key，chmod，追加 authorized_keys，验证。

可能收益：
- SSH key 任务实用。

风险：
- 说“任何多步骤命令都不要和 `>` 或 `>>` 在同一行组合”过强，很多文件任务必须重定向。
- 与 S07/S16 等策略冲突。

建议：
- P2/P1。保留 SSH 部分，删除“任何任务都不要重定向组合”的泛化。

### S24 完成前综合验证策略

作用：
- 完成前检查文件存在、内容、权限。

可能收益：
- 通用且合理。

风险：
- 若被 after_round hook 强制化，会造成重复验证。
- 权限示例偏 SSH key 任务。

建议：
- P2。适合作为轻量通用策略，但应避免每轮重复注入。

### S25 devsecops `.env` 文件创建策略

作用：
- 固定创建 `.env.template`、`.env.secure`、`.gitignore`、`load_env.sh`。

可能收益：
- 对某个 devsecops task 有用。

风险：
- 路径、密钥内容、文件名全部硬编码。
- 可能生成敏感-looking 内容，误导非相关任务。

建议：
- P1。任务特化，不能全局注入。

### S26 CSV-to-JSON overheat devices 策略

作用：
- 固定生成 `overheat_devices.json` 和 `.log`。

可能收益：
- 对具体 CSV 过滤任务有用。

风险：
- 输出记录完全硬编码，泛化性极差。

建议：
- P1。只适合检索命中特定任务时启用。

### S27 `Are you sure` prompt 后系统监控任务综合验证策略

作用：
- 固定 cat `thresholds.conf`、`monitor_demo.sh`、`alert.log`。

可能收益：
- 对具体 sys_monitoring 任务有用。

风险：
- 文件路径完全硬编码。
- 会让其他任务在 completion prompt 后执行错误验证。

建议：
- P1。任务特化。

### S28 所有文件创建完成后的完成前验证策略

作用：
- 创建全部文件后运行单次综合验证，验证通过即完成。

可能收益：
- 通用方向正确。

风险：
- “所有文件创建完成”需要 agent 自己判断，可能误判。
- 与 S24/S18 重复。

建议：
- P2。可与 S24 合并为一条通用 completion hygiene 策略。

### S29 log/structured text 解析策略

作用：
- 先 `head -5` 看格式，再用单个成熟命令解析。
- 失败后换方法，不要微调同一个失败命令。

可能收益：
- 通用价值较高，适合 log parsing 任务。

风险：
- 示例 `awk -F"` 不完整，可能被模型误抄。

建议：
- P3/P2。值得保留，修正示例即可。

### S30 多输出文件 + packaging 策略

作用：
- 先建 reports，再分别验证两个输出文件，最后打 tarball。

可能收益：
- 对报告打包类任务有帮助。

风险：
- `status_per_hour.csv`、`top5_endpoints.txt` 等特化。
- “分别验证每个文件”与“只做一次综合验证”策略冲突。

建议：
- P2。保留为报告打包类检索策略。

### S31 archive creation + extraction verification 策略

作用：
- 创建 `/home/user/archives/src_backup_2023.tar.gz`，解压到 `src_restored`，写 compression log，diff 文件。

可能收益：
- 对某个 src backup 任务有用。

风险：
- 路径和文件名全部硬编码。

建议：
- P1。任务特化。

### S32 context warning 后合并验证策略

作用：
- context warning 出现后，用一条 `test -f && diff && cat` 做综合验证。

可能收益：
- 对 S31 那类 archive 任务有帮助。

风险：
- 仍然硬编码 `/home/user/archives/src_backup_2023.tar.gz` 等路径。
- 只适合特定任务。

建议：
- P1。任务特化，不能全局注入。

## Hook 分析

### H01 `before_llm_call_candidate_0_hookgrp_pre_action_controls_e8f92ae8.py`

Hook point：`before_llm_call`

作用：
- 如果最近命令中 `/home/user/` 路径上的 `cat` 达到 3 次，向 prompt 追加 `[WARN]`，建议一次综合检查。
- 如果最近 `echo/printf` 命令达到 5 次，追加 `[WARN]`，建议单条 printf。

可能收益：
- 对重复验证和碎片化文件创建有一定抑制。

风险：
- 硬编码 `/home/user/`，在 `/app` 等工作目录下失效。
- 只看命令数量，不判断是否同一文件或是否必要。
- 对需要逐步构建文件的任务可能过早施压。

建议：
- P2。可保留但要改为基于路径/文件归一化后的重复检测。

### H02 `before_llm_call_candidate_1_hookgrp_pre_action_controls_0bbc3050.py`

Hook point：`before_llm_call`

作用：
- 统计 LLM 调用次数，超过 3 次后追加 `[GUIDANCE]`，要求输出合法 JSON。
- 检测 last_analysis 中是否出现 `Are you sure` 或 `verify`，试图提醒不要重复验证。

可能收益：
- 本意是减少解析错误和 verification loop。

致命风险：
- 注入文本要求“exactly these fields: `analysis`, `plan`, `commands`”，漏掉 `task_complete`。
- Terminus2 需要 `task_complete` 才能正常完成；这会让模型不敢或忘记提交完成信号。
- `verification_attempts` 只有在已经 `>2` 时才递增，逻辑上几乎不会启动。

建议：
- P0。删除或重写。若保留，必须包含 `task_complete`，且不能用“exactly fields”压制 runtime schema。

### H03 `before_llm_call_candidate_1_hookgrp_pre_action_controls_3ddce9d2.py`

Hook point：`before_llm_call`

作用：
- 如果 `verification_attempts >= 2`，追加 debug 提醒。
- 如果 `task_complete_attempts >= 2`，追加 debug 提醒。

可能收益：
- 轻量提醒，理论上可帮助避免循环。

风险：
- 这两个计数器在本 hook 内不递增，依赖其他 hook 写入；实际可能永远不触发。
- `[DEBUG]` 标签会污染 prompt，收益有限。

建议：
- P3。低风险但价值很低，建议删除或并入一个统一 loop monitor。

### H04 `before_llm_call_candidate_1_hookgrp_pre_action_controls_b871ce71.py`

Hook point：`before_llm_call`

作用：
- 每超过 5 次调用，将 prompt 中的 `analysis:`、`plan:`、`commands:` 全文替换成 Markdown 标题。
- 追加 `[CRITICAL CONSTRAINTS]` 和 `[PREVIOUS APPROACH FAILED]`，要求 JSON only、短分析、少命令、用 printf 替代 heredoc。

可能收益：
- 试图压缩输出、减少解析错误。

严重风险：
- `str.replace` 是全文替换，会改坏任务描述、终端输出、错误反馈里的普通文本。
- `[PREVIOUS APPROACH FAILED]` 可能是假的，会误导模型放弃正确路线。
- 注入的命令列表示例含换行，反而可能制造 JSON escaping 问题。
- 强制不用 heredoc 对长文件创建任务有害。

建议：
- P1/P0。建议删除。若需要 JSON 修复，应在 parser 层处理，不应粗暴改 prompt。

### H05 `before_llm_call_candidate_1_hookgrp_pre_action_controls_d2d59c8d.py`

Hook point：`before_llm_call`

作用：
- 从第 6 次左右 LLM 调用开始追加 `[CONTEXT CONSTRAINT]`。
- 附带 `context.original_instruction[:150]` 作为 Task 摘要，并建议简单单行命令、避免复杂 heredoc。

可能收益：
- 如果 `original_instruction` 是真实任务，短提醒有一定价值。

严重风险：
- 旧注入顺序中，`original_instruction` 前部是 `[LEARNED STRATEGIES]`，导致 Task 摘要变成策略文本。
- 即使当前代码已改成“先任务后策略”，150 字截断也可能漏掉关键约束。
- “避免复杂 heredoc”过度泛化。

建议：
- P1。保留前必须确保 HookContext 有独立 `task_instruction` 字段，而不是截 `original_instruction`。

### H06 `before_llm_call_candidate_1_hookgrp_pre_action_controls_e1888498.py`

Hook point：`before_llm_call`

作用：
- 第 0 轮如果原始任务包含 `dpkg`，提醒必须重定向到文件。
- 如果包含 `checksum`，提醒 checksum log 用两个空格分隔。

可能收益：
- 对 dpkg/checksum 任务有明确帮助。

风险：
- 特化逻辑，可能对非相关任务无效但一般不破坏。
- checksum 格式并非所有任务都要求两个空格。

建议：
- P2/P3。可保留，但只应关键词触发，且不要作为全局常驻 hook。

### H07 `before_llm_call_s10_candidate_0_hookgrp_pre_action_controls_1506fcd9.py`

Hook point：`before_llm_call`

作用：
- 从 prompt 中扫描 `cat`/`ls` 行，记录验证历史。
- 第 10 次调用后追加 `[FOCUS]`，附带 `context.original_instruction[:150]`。

可能收益：
- 试图减少重复验证，并在长交互时拉回任务焦点。

严重风险：
- 重复检测逻辑几乎无效：`last_3` 在 append 前计算，且 `all(... for s in [last_3[-1]])` 只是检查最后一项在前三项里。
- `[FOCUS]` 同样依赖 `original_instruction[:150]`，旧策略前置时会截到策略而非任务。
- 第 10 轮后每次都会注入，可能造成 prompt 噪声。

建议：
- P1。删除或重写为基于 `context.last_commands` 的真实重复检测。

### H08 `before_llm_call_s14_candidate_0_hookgrp_pre_action_controls_2e0d547d.py`

Hook point：`before_llm_call`

作用：
- 如果 `context.is_task_complete` 为真，增加 verify_count，超过 3 次提醒停止验证。
- 如果 `completion_attempts >= 2`，追加 `[RESTART]` 并截取原始任务前 500 字。

可能收益：
- 想在完成后避免继续验证。

风险：
- `context.is_task_complete` 是否在 before_llm_call 阶段可靠不确定。
- `completion_attempts` 本 hook 不递增，依赖其他 hook。
- `[RESTART]` 可能把已经接近完成的任务拉回重读，造成浪费。

建议：
- P2。逻辑不稳定，建议删除或迁移到 after_round 的统一 completion controller。

### H09 `before_execute_candidate_0_hookgrp_pre_action_controls_1c7ba974.py`

Hook point：`before_execute`

作用：
- 统计 `ls/find/dir`，超过 3 次后直接丢弃后续同类命令。
- 统计 `cat/head/grep` 验证次数。

可能收益：
- 减少表面检查和重复验证。

严重风险：
- 静默丢弃命令会让 agent 以为命令执行了，但没有真实输出。
- `find`/`ls` 在探索未知文件结构时是必要动作。
- 计数器跨轮积累，可能在后续关键步骤误杀命令。

建议：
- P1。不要静默丢弃命令。最多向 prompt 提醒。

### H10 `before_execute_candidate_1_hookgrp_pre_action_controls_022ffa35.py`

Hook point：`before_execute`

作用：
- 首次看到文件操作命令时，将该命令改写为 `su - user -c "<cmd>"`。

可能收益：
- 试图解决 root/user 权限不一致问题。

致命风险：
- terminal-bench 环境中未必存在 `user` 用户；历史轨迹已见 `user does not exist`。
- 双引号嵌套会破坏很多命令。
- 只改第一条文件操作，行为不可预测。

建议：
- P0。删除。用户切换必须先检测用户存在，且不应由 hook 自动包命令。

### H11 `before_execute_candidate_1_hookgrp_pre_action_controls_30f1f9d1.py`

Hook point：`before_execute`

作用：
- 试图把相对路径转换为绝对路径。
- 对包含 `/` 且不是绝对路径的行拼接 `/home/user`。

可能收益：
- 本意是减少路径解析失败。

严重风险：
- 会把 `./configure`、`../foo`、脚本路径、命令参数等误改。
- 硬编码 `/home/user`，而很多任务工作目录是 `/app`。
- 按整行处理，不理解 shell 语法。

建议：
- P0/P1。删除。路径修正应由模型基于 `pwd`/任务路径判断。

### H12 `before_execute_candidate_1_hookgrp_pre_action_controls_77619f48.py`

Hook point：`before_execute`

作用：
- 连续验证命令达到 2 条时丢弃后续验证命令。
- 尝试丢弃过于复杂的 heredoc 多行命令。

可能收益：
- 减少重复验证。

严重风险：
- “连续两条验证命令”在复杂任务中很正常，例如 `ls` 后 `cat`。
- 第二段 complex command drop 逻辑有 bug：只有 `complex_cmd_count >= 2` 时才递增，初始永远不增长。
- 静默丢弃命令仍然会让 agent 状态错乱。

建议：
- P1。删除或改为非阻断提醒。

### H13 `before_execute_candidate_1_hookgrp_pre_action_controls_c597eb6c.py`

Hook point：`before_execute`

作用：
- 过滤空命令。
- 如果命令含 `$(` 或反引号，尝试替换某些 shell expansion。
- 将超长 duration 限制到 30 秒。
- 命令数过多时写 `stop_loop` 到 kv。

可能收益：
- 过滤空命令和限制超时有一点价值。

严重风险：
- `keystrokes.replace('$(', '(')` 会破坏合法 command substitution。
- `stop_loop` 只是写 kv，没有任何地方消费。
- 将 duration 强制降到 30 秒可能杀掉本来需要更久的安装/测试命令。

建议：
- P1。只保留空命令过滤；不要改写 shell 语法。

### H14 `before_execute_candidate_1_hookgrp_pre_action_controls_de3fac5a.py`

Hook point：`before_execute`

作用：
- 过滤空命令。
- 如果多次执行且命令包含 `&&` 或 `||`，只记录 `complex_cmd_warning`。

可能收益：
- 空命令过滤合理。

风险：
- 只记录 kv，不反馈给 agent，实际价值有限。
- `&&` / `||` 是常见且有用的 shell 模式，不应天然视作问题。

建议：
- P3。低风险，可保留空命令过滤，删除无用计数。

### H15 `before_execute_s10_candidate_0_hookgrp_pre_action_controls_a846437a.py`

Hook point：`before_execute`

作用：
- 对完全重复的 `cat/ls/head` 命令去重。
- 对某些 `find` 命令前置 `mkdir -p /home/user/audit &&`。
- 限制 duration 到 30 秒。

可能收益：
- 完全重复验证去重有一定价值。

严重风险：
- 给 `find` 前面插入 `mkdir -p /home/user/audit` 没有通用依据，会污染文件系统。
- duration 限制可能造成误超时。
- 只在同一批 commands 内去重，跨轮重复检测不足。

建议：
- P1。删除自动插 mkdir；只保留“同批完全重复命令去重”也要谨慎。

### H16 `before_execute_s14_candidate_0_hookgrp_pre_action_controls_91619836.py`

Hook point：`before_execute`

作用：
- 过滤空命令。
- 丢弃以 `>` 结尾的命令。
- 丢弃超过 3 行的命令。
- 第 2 次 heredoc 后在命令前拼注释。

可能收益：
- 丢弃明显残缺的 trailing redirection 命令有一定价值。

致命风险：
- 很多合法文件创建需要超过 3 行，直接丢弃会阻断任务。
- 在 heredoc 命令前拼接注释可能破坏 shell 结构。
- trailing redirection 判断访问 `s.split('\n')[-2]`，单行命令可能触发索引问题。

建议：
- P0/P1。删除。命令完整性校验应在 parser 层做，不应粗暴丢弃。

### H17 `after_execute_candidate_0_hookgrp_post_action_controls_4d7741ab.py`

Hook point：`after_execute`

作用：
- 根据 terminal output 记录 file creation、shebang missing、heredoc issues、parse issues、sync_verified。
- 不直接修改输出。

可能收益：
- 作为状态收集器有一定价值，可供 after_round 使用。

风险：
- 用 terminal output 字符串判断命令意图，容易误判。
- shebang 检测条件过于特化：`.sh` + `echo` + `sudo ufw`。
- `heredoc_issues` 列表混入 episode 和字符串，类型不一致。

建议：
- P2/P3。低直接破坏性，但信号质量差。

### H18 `after_execute_candidate_0_hookgrp_post_action_controls_d909fbbb.py`

Hook point：`after_execute`

作用：
- 记录 recent output。
- 如果输出含 `>`、以 `...` 结尾或很短，认为 terminal stuck，并可能追加 `[WARNING]`。
- 记录文件创建、错误信息。
- 超过 3000 字且 50 行以上时截断输出。

可能收益：
- 长输出截断对上下文有帮助。

严重风险：
- `>` 在 shell 输出、重定向、prompt、文本内容里很常见，误判 stuck 概率高。
- `recent = recent[-5:]` 没写回 kv，裁剪无效。
- `if creator in terminal_output and 'Created' in terminal_output or 'success' ...` 优先级导致只要 output 有 success 就记录。
- 截断会改变 agent 看到的真实终端反馈。

建议：
- P1/P2。只保留更安全的长度截断，并避免追加拟人化 warning 到终端输出。

### H19 `after_execute_candidate_1_hookgrp_post_action_controls_082538f5.py`

Hook point：`after_execute`

作用：
- 超过 4000 字时截断输出。
- 输出行数超过 10 时记录 `multi_line_output`。
- 记录 `Permission denied` / `command not found`。

可能收益：
- 长输出截断和错误计数有一定价值。

风险：
- 注释写明 “Task 944”，说明来自单任务。
- `context_savings += len(terminal_output) - 4000` 在截断后通常为负数，统计错误。
- `cmd_parse_flags` 只增不清，会无限增长。

建议：
- P2。可重写为通用 output compressor。

### H20 `after_execute_candidate_1_hookgrp_post_action_controls_321b92b4.py`

Hook point：`after_execute`

作用：
- 如果 output 包含 `heredoc` 或 `EOF`，累计 heredoc attempts。
- 超过 2 次后追加 `[WARN]`，建议 echo/printf。

可能收益：
- 对确实反复 heredoc 失败的任务有帮助。

风险：
- `EOF` 可能出现在正常文件内容或错误消息中。
- 直接向 terminal output 追加文本，会让模型误以为这是命令真实输出。

建议：
- P2。若保留，应通过 next_prompt 提醒，而不是伪造 terminal output。

### H21 `after_execute_candidate_1_hookgrp_post_action_controls_a23899ac.py`

Hook point：`after_execute`

作用：
- 尝试追踪 log 写入内容和 log write history。
- 如果最近 3 条 echo/printf 完全相同，向 terminal output 追加 `[LOOP DETECTED]`。

可能收益：
- 可以发现重复写日志循环。

风险：
- `'write' in context.last_commands` 比较 list 与字符串，基本不成立。
- `history = history[-20:]` 没写回 kv。
- 仍然是向 terminal output 注入非真实文本。

建议：
- P2/P3。实现 bug 较多，建议删除或重写。

### H22 `after_execute_candidate_1_hookgrp_post_action_controls_d5b1ecce.py`

Hook point：`after_execute`

作用：
- 超过 3000 字时截断输出，保留前 1000 行和后 1000 行。
- 如果输出首行像验证命令，增加 verify_count。
- verify_count > 3 时写 loop_warning 到 kv。

可能收益：
- 输出截断方向有价值。

风险：
- 1000 行 + 1000 行仍可能非常长。
- 通过 terminal output 首行判断命令类型不可靠。
- loop_warning 写入 kv 但不一定被其他 hook 消费。

建议：
- P2。改成固定字符预算截断更稳。

### H23 `after_execute_candidate_1_hookgrp_post_action_controls_e7ace1fd.py`

Hook point：`after_execute`

作用：
- 检测 output 中是否出现 `cat/ls/wc/grep/file` 等。
- 如果检测到重复验证，向 terminal output 追加 `[GUIDANCE]`，含 ANSI 黄色转义。

可能收益：
- 试图打断重复验证。

严重风险：
- `detected in recent` 比较 list 是否在 list 中，逻辑不符合“重复命令集合”的预期。
- ANSI 转义码会污染文本。
- 向 terminal output 注入非真实内容。

建议：
- P1/P2。删除或改成普通 next_prompt 提醒。

### H24 `after_execute_candidate_1_hookgrp_post_action_controls_e8ed3dd8.py`

Hook point：`after_execute`

作用：
- 如果 output 包含 `su` 和 cron 相关内容，尝试提取 cron entry。
- 若未看到 `crontab -l`，追加 `[VERIFY]` 建议验证。

可能收益：
- 对 cron 任务有帮助。

风险：
- 任务特化，且依赖 output 中出现 `CRON_ENTRY`。
- 向 terminal output 注入文本。
- 对用 `su` 的非 cron 任务可能误触发。

建议：
- P2。只应在 cron 任务中启用。

### H25 `after_round_candidate_0_hookgrp_post_action_controls_1fab62c7.py`

Hook point：`after_round`

作用：
- 根据 kv 中的 `shebang_missing`、`heredoc_issues`、`parse_issues`、命令数量等追加 next_prompt。
- 首次 `is_task_complete` 时发起 `request_new_turn: True`，要求额外验证。

可能收益：
- 聚合 after_execute 状态，进行针对性提醒。

致命风险：
- 首次完成强制续轮，会和其他 after_round hook 叠加，导致完成后停不下来。
- `CONTEXT WARNING` 在命令数多时每轮都可能触发。
- 依赖上游粗糙 kv 信号，误报后会持续污染 prompt。

建议：
- P0/P1。删除 completion 强制续轮；其他提示也应有 once-only 限制。

### H26 `after_round_candidate_0_hookgrp_post_action_controls_8d872b17.py`

Hook point：`after_round`

作用：
- completion 时，如果原始任务包含 `timeout` 或 `log_level`，追加 `[VERIFY]` completion prompt。

可能收益：
- 对某类误导验证任务可能有用。

风险：
- 不发 `request_new_turn`，但 next_prompt 可能仍加入下一轮提示。
- 如果真实任务确实包含 timeout/log_level，这条可能造成重复确认。

建议：
- P2/P3。保留价值有限，建议任务检索启用。

### H27 `after_round_candidate_0_hookgrp_post_action_controls_c4d0b384.py`

Hook point：`after_round`

作用：
- 记录 command_history。
- 检测 verification loop、stuck state、接近 episode 上限、首次 completion、解析错误。
- 多处返回 next_prompt，部分带 `request_new_turn: True`。

可能收益：
- 试图做综合 loop controller。

严重风险：
- `cmd in ['cat', 'ls -la', ...]` 只匹配完整字符串，很多真实命令不命中。
- stuck_count 来自其他 hook 的粗糙判断，误报会强制续轮。
- near_limit_warned 写在 return 后面，永远不会设置，导致可重复提示。
- 首次 completion 又追加验证，与 H25/H29/H31 重复。
- `error` / `command` 子串检测过宽，正常输出也可能触发 parsing issues。

建议：
- P0/P1。不要与其他 after_round completion hook 并存；如果要保留，需重写为唯一的 after_round controller。

### H28 `after_round_candidate_1_hookgrp_post_action_controls_29d55b1a.py`

Hook point：`after_round`

作用：
- 如果多次 task_complete，追加 `[REFOCUS]`，要求回看原任务。
- 检测 `timeout (TO)` / `log_level (LOG_LEVEL)`。

可能收益：
- 想处理 repeated completion。

严重风险：
- 使用 `context.episode.original_instruction`，但 `context.episode` 是数字，不是对象；触发时会抛 AttributeError。
- `complete_count` 增加后没有写回 kv，逻辑实际上不生效。

建议：
- P1。删除，当前实现有明显 bug。

### H29 `after_round_candidate_1_hookgrp_post_action_controls_3151307f.py`

Hook point：`after_round`

作用：
- 首次 `is_task_complete` 时 `request_new_turn: True`，要求验证。
- 第二次 completion 时允许结束。

可能收益：
- 单独看是“两步确认”机制。

致命风险：
- 与其他多个 completion hook 并存时，第一次完成会被多个 hook 同时拦截。
- 若 H02 让模型不输出 `task_complete`，这个 hook 的状态机也无法正常推进。
- 对已经验证过的任务仍强制多一轮，增加成本和超时风险。

建议：
- P0/P1。最多只能有一个全局 completion controller，且必须检查最近是否已经做过验证。

### H30 `after_round_candidate_1_hookgrp_post_action_controls_77441313.py`

Hook point：`after_round`

作用：
- completion 时累计 completion_attempts / verify_attempts。
- verify_attempts >= 2 时 `request_new_turn: True`，注入 `[LOOP DETECTED]`。
- complex_cmd_count > 3 时注入 warning。

可能收益：
- 想打断 completion/verification loop。

严重风险：
- loop 已经发生时继续 `request_new_turn`，可能反而延长循环。
- verify_attempts 是每次 completion 增加，而不是真实验证动作。
- `inject` 与 `next_prompt` 协议混用，依赖 runtime merge 行为。

建议：
- P1。删除强制续轮；如果检测到 loop，应允许结束或给一次非阻断提示。

### H31 `after_round_candidate_1_hookgrp_post_action_controls_b1215fca.py`

Hook point：`after_round`

作用：
- verify_count > 3 且未完成时注入 `[GUIDANCE]`。
- 首次 completion 时 `request_new_turn: True`，要求一次验证。
- 检测重复命令，注入 `[LOOP DETECTED]`。

可能收益：
- 想做验证循环防护。

致命风险：
- 又一个首次 completion 拦截器，与 H25/H27/H29 重复。
- 第二次 completion 会把 complete_count 重置为 0，后续 completion 可能再次被当作首次。
- `verify_count` 来源不稳定，可能被 before/after_execute hook 错误累积。

建议：
- P0/P1。不要和其他 after_round completion hook 并存。

### H32 `after_round_s14_candidate_0_hookgrp_post_action_controls_5943530c.py`

Hook point：`after_round`

作用：
- 检测验证命令，达到 3 次时 `request_new_turn: True` 并注入 `[LOOP DETECTED]`。
- 如果命令中 `>` 超过 2 个，注入 `[COMMAND ERROR]` 并续轮。

可能收益：
- 试图识别重复验证和复杂重定向。

严重风险：
- `any(cmd in context.last_commands for cmd in verify_cmds)` 是把字符串查 list 元素，通常不符合预期；如果 last_commands 是完整命令，则 `cat` 不等于 `cat file`。
- `verification_count` 本地变量递增后没有写回 kv，下轮丢失。
- 复杂重定向不一定错误，例如 heredoc 或多文件生成。
- 又一个 `request_new_turn` 来源。

建议：
- P1。删除或重写。

### H33 `on_timeout_candidate_0_hookgrp_pre_action_controls_a92a297b.py`

Hook point：`on_timeout`

作用：
- 如果超时命令含 `printf`，在输出后追加 `[TIMEOUT]`，建议使用 `sed -i`。

可能收益：
- 几乎没有稳定收益。

风险：
- 把 printf 超时错误归因于 printf 本身，这是错误泛化。
- `sed -i` 并不能替代文件创建或多行写入。

建议：
- P1。删除。

### H34 `on_timeout_candidate_1_hookgrp_pre_action_controls_0672a5c8.py`

Hook point：`on_timeout`

作用：
- 记录 timeout_count。
- 若命令长度超过 100，在 kv 里写 long_cmd_warning。

可能收益：
- 作为统计信号有一点用。

风险：
- 不返回 warning，agent 看不到。
- 长命令不一定是超时原因。

建议：
- P3。低风险但实际无效。

### H35 `on_timeout_candidate_1_hookgrp_pre_action_controls_274f5346.py`

Hook point：`on_timeout`

作用：
- 第二次及以后 timeout，在输出后追加 `[TIMEOUT WARNING]`，建议简化命令。

可能收益：
- 对连续超时有轻量帮助。

风险：
- 所有 timeout 都归因于命令复杂/长运行，可能误导安装、测试、服务启动类任务。

建议：
- P2。可以保留，但提示应更中性。

### H36 `on_timeout_candidate_1_hookgrp_pre_action_controls_e2ba4037.py`

Hook point：`on_timeout`

作用：
- 每次 timeout 都追加 `[TIMEOUT] Command timed out...`。

可能收益：
- 明确告诉 agent 上一步超时。

风险：
- runtime 通常已经知道 timeout；重复提示会增加噪声。
- “missing proper syntax” 未必成立。

建议：
- P2/P3。可保留一个 timeout 提示，但不要多个重复 hook。

### H37 `on_timeout_s14_candidate_0_hookgrp_pre_action_controls_be501739.py`

Hook point：`on_timeout`

作用：
- 每次 timeout 都追加 `[TIMEOUT] This command was too long or complex...`。

可能收益：
- 和 H36 类似。

风险：
- 与 H36 重复。
- 仍然把 timeout 简化为“命令太长或复杂”。

建议：
- P2/P3。若保留 timeout hook，只留 H36/H37 之一并改成中性提示。

## 按风险建议删除/保留

### 建议优先删除的 P0/P1 Hook

- H02 `before_llm_call_candidate_1_hookgrp_pre_action_controls_0bbc3050.py`
- H04 `before_llm_call_candidate_1_hookgrp_pre_action_controls_b871ce71.py`
- H05 `before_llm_call_candidate_1_hookgrp_pre_action_controls_d2d59c8d.py`
- H07 `before_llm_call_s10_candidate_0_hookgrp_pre_action_controls_1506fcd9.py`
- H09 `before_execute_candidate_0_hookgrp_pre_action_controls_1c7ba974.py`
- H10 `before_execute_candidate_1_hookgrp_pre_action_controls_022ffa35.py`
- H11 `before_execute_candidate_1_hookgrp_pre_action_controls_30f1f9d1.py`
- H12 `before_execute_candidate_1_hookgrp_pre_action_controls_77619f48.py`
- H13 `before_execute_candidate_1_hookgrp_pre_action_controls_c597eb6c.py`
- H15 `before_execute_s10_candidate_0_hookgrp_pre_action_controls_a846437a.py`
- H16 `before_execute_s14_candidate_0_hookgrp_pre_action_controls_91619836.py`
- H18 `after_execute_candidate_0_hookgrp_post_action_controls_d909fbbb.py`
- H23 `after_execute_candidate_1_hookgrp_post_action_controls_e7ace1fd.py`
- H25 `after_round_candidate_0_hookgrp_post_action_controls_1fab62c7.py`
- H27 `after_round_candidate_0_hookgrp_post_action_controls_c4d0b384.py`
- H28 `after_round_candidate_1_hookgrp_post_action_controls_29d55b1a.py`
- H29 `after_round_candidate_1_hookgrp_post_action_controls_3151307f.py`
- H30 `after_round_candidate_1_hookgrp_post_action_controls_77441313.py`
- H31 `after_round_candidate_1_hookgrp_post_action_controls_b1215fca.py`
- H32 `after_round_s14_candidate_0_hookgrp_post_action_controls_5943530c.py`
- H33 `on_timeout_candidate_0_hookgrp_pre_action_controls_a92a297b.py`

### 相对可保留但应重写/限域的 Hook

- H01：重复 cat / echo 监控，但要去掉 `/home/user` 硬编码。
- H06：dpkg/checksum 首轮提示，但只在任务关键词命中时启用。
- H14：空命令过滤，其他计数可删。
- H17：状态收集器可保留，但要修正信号质量。
- H19/H22：长输出截断可保留，改成统一 output compressor。
- H20/H21/H24：只在对应 heredoc/log/cron 任务中启用，且不要伪造 terminal output。
- H26：timeout/log_level 特化，最好改成策略而不是 hook。
- H34/H35/H36/H37：timeout hook 只留一个中性版本。

## 推荐的清理方向

### Strategy

建议把 32 条策略分成两层：

1. 全局通用策略，最多 5 条：
   - 完成前做一次综合验证。
   - 写脚本/多行文件时根据内容长度选择 printf 或 heredoc，不绝对禁止。
   - Makefile recipe 必须 tab。
   - log/structured text 先观察格式再解析。
   - 避免无意义重复验证。

2. 任务检索策略：
   - archive/dpkg/DNS/SSH/Docker/CSV/devsecops/sys_monitoring 等全部只有在任务描述强匹配时注入。

### Hook

建议把 active hook 压缩到 3 类以内：

1. 只读观测 hook：记录命令、输出长度、错误类型，不直接改行为。
2. 输出截断 hook：只做安全的字符预算压缩。
3. 唯一 after_round controller：最多允许一次验证续轮，而且必须检查最近是否已经验证过；不能多个 hook 同时 request_new_turn。

不要再启用：
- 自动改写 shell 命令的 hook。
- 静默丢弃命令的 hook。
- 会漏掉 `task_complete` 的 JSON 指导。
- 多个 completion 拦截器并存。
- 向 terminal output 追加非真实内容的 hook。

## 结论

这批 patch 里确实有一些从失败轨迹中学到的合理经验，但当前形态更像“未过滤的经验堆叠”。对于 terminal-bench 这种多任务评测，全局注入强任务特化策略和强行为 hook 会显著增加退化概率。

最关键的修复不是微调某一条文案，而是改变 patch 采纳标准：

- 策略必须可检索、可去重、可限域。
- Hook 默认不能修改命令，只能观察或软提示。
- `after_round` 必须单控制器化，不能多个 hook 独立强制续轮。
- Canary 必须把 episode length、timeout、parse error、completion loop 作为拒绝指标，而不只看平均 reward。

