# SkyRL 元学习训练使用手册

## 1. 快速开始

### 1.1 启动训练

```bash
# 确保先清理残留（见第 3 节），然后：
bash /home/ray/SkyRL/examples/train_integrations/harbor/run_megatron.sh 2>&1 | tee /home/ray/train.log
```

### 1.2 关键参数说明

在 `run_megatron.sh` 中修改以下参数：

| 参数 | 含义 | 建议值 |
|------|------|--------|
| `RUN_NAME` | 实验名称，决定所有输出目录 | 每次实验换一个新名字 |
| `harbor_trial_config.agent.name` | 使用的智能体 | `terminus-1` 或 `terminus-2` |
| `generator.n_samples_per_prompt` | 每个任务的轨迹数 | 4-8 |
| `trainer.train_batch_size` | 每个 batch 的任务数 × 轨迹数 | 16（4任务×4轨迹） |
| `generator.meta.enabled` | 是否启用元学习 | `true` |
| `generator.meta.interval_batches` | 每隔几个 batch 做一次元学习 | 1-4 |
| `generator.meta.max_candidates` | 每次生成几个补丁候选 | 2-4 |
| `generator.meta.canary_num_tasks` | 金丝雀评估的任务数 | 8-32 |
| `generator.meta.canary_n_samples` | 金丝雀评估每个任务的轨迹数 | 4-8 |
| `generator.meta.override_base` | 元学习补丁目录 | 按智能体选择（见下） |

**补丁目录**：
- terminus-1：`/home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus1`
- terminus-2：`/home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus2`

### 1.3 切换智能体

切换智能体需要同时修改两处：

```bash
# 使用 terminus-1
harbor_trial_config.agent.name=terminus-1
generator.meta.override_base=.../meta_patches/terminus1

# 使用 terminus-2
harbor_trial_config.agent.name=terminus-2
generator.meta.override_base=.../meta_patches/terminus2
```

## 2. 产出文件说明

训练过程中会在以下位置产生文件：

### 2.1 元学习补丁目录（最重要，会影响下次训练）

```
meta_patches/<agent>/
├── .git/                    # Git 版本控制
├── .gitignore
├── hooks/                   # 模型生成的 Python hook 代码
│   ├── before_llm_call.py
│   ├── after_execute.py
│   ├── before_execute.py
│   ├── on_timeout.py
│   ├── on_parse_error.py
│   └── *.py.bak             # 快照备份
├── _templates/              # 模型修改的智能体模板文件
│   └── *.txt
├── *.yaml                   # YAML 配置覆盖
│   ├── planner_overrides.yaml
│   ├── retry_overrides.yaml
│   ├── finish_overrides.yaml
│   ├── verification_overrides.yaml
│   ├── system_prompt_overrides.yaml
│   └── behavior_policy.yaml
```

**这是最关键的目录**。训练中被接受的补丁会永久写入这里。如果不清理，下次训练会以这些补丁为基线，可能导致性能异常。

### 2.2 元学习日志目录

```
/tmp/skyrl-logs/
├── meta_run_<YYYYMMDD_HHMMSS>/    # 每次运行的独立目录
│   ├── diagnosis_conversations.jsonl   # 诊断器 LLM 对话记录
│   ├── planning_conversations.jsonl    # 规划器 LLM 对话记录
│   ├── cycle_summary.log              # 每轮元学习的摘要
│   └── canary_traces/                 # 金丝雀评估轨迹
├── diagnosis_history.jsonl    # 全局诊断历史（旧格式，可能存在）
└── meta_conversations.jsonl   # 全局对话记录（旧格式，可能存在）
```

### 2.3 训练输出目录

```
/home/test/test1714/wxh/skyrl/<RUN_NAME>/
├── trials_run/     # Harbor 试验执行目录（智能体轨迹）
└── ckpts/          # 模型 checkpoint

/tmp/skyrl-logs/<RUN_NAME>/    # SkyRL 训练日志

~/train.log                    # 主训练日志（tee 输出）
```

### 2.4 Wandb

训练指标会上传到 Wandb 项目 `meta`，run 名称为 `RUN_NAME`。

## 3. 清理残留文件

### 3.1 必须清理：元学习补丁目录

**每次重新开始实验前必须执行**。上次训练接受的补丁会留在磁盘上，影响新实验的基线。

```bash
# === 清理 terminus-1 补丁 ===
cd /home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus1
rm -f *.yaml
rm -rf hooks/*.py hooks/*.bak _templates
git add -A && git commit -m "clean: reset for new experiment"

# === 清理 terminus-2 补丁 ===
cd /home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus2
rm -f *.yaml
rm -rf hooks/*.py hooks/*.bak _templates
git add -A && git commit -m "clean: reset for new experiment"
```

验证清理结果：
```bash
# 应该只看到 .git/ .gitignore hooks/（空目录）
ls -la /home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus1/
ls -la /home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus1/hooks/
```

### 3.2 可选清理：元学习日志

这些日志不影响训练结果，但会占用磁盘空间。

```bash
# 删除所有历史元学习日志
rm -rf /tmp/skyrl-logs/meta_run_*

# 删除旧格式的全局日志
rm -f /tmp/skyrl-logs/diagnosis_history.jsonl
rm -f /tmp/skyrl-logs/meta_conversations.jsonl
rm -rf /tmp/skyrl-logs/accepted_patches
```

### 3.3 可选清理：训练输出

如果要完全从头开始（不恢复 checkpoint）：

```bash
# 删除特定实验的 trials 和 checkpoints
# 注意：请替换 <RUN_NAME> 为实际的实验名
rm -rf /home/test/test1714/wxh/skyrl/<RUN_NAME>/

# 删除训练日志
rm -rf /tmp/skyrl-logs/<RUN_NAME>/

# 清空主日志
> /home/ray/train.log
```

**注意**：如果 `run_megatron.sh` 中设置了 `trainer.resume_mode=latest`，SkyRL 会尝试从 checkpoint 恢复。如果你想从头训练，要么删除 checkpoint 目录，要么换一个新的 `RUN_NAME`。

### 3.4 一键清理脚本

以下命令清理当前智能体的补丁目录和所有元学习日志：

```bash
# 一键清理（terminus-1）
AGENT=terminus1 && \
cd /home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/$AGENT && \
rm -f *.yaml && rm -rf hooks/*.py hooks/*.bak _templates && \
git add -A && git commit -m "clean: reset for new experiment" --allow-empty && \
rm -rf /tmp/skyrl-logs/meta_run_* && \
rm -f /tmp/skyrl-logs/diagnosis_history.jsonl /tmp/skyrl-logs/meta_conversations.jsonl && \
echo "Done: $AGENT patches and meta logs cleaned"
```

把 `AGENT=terminus1` 改为 `AGENT=terminus2` 即可清理 terminus-2。

## 4. 查看训练状态

### 4.1 实时日志

```bash
# 主训练日志
tail -f /home/ray/train.log

# 查找元学习相关输出
grep -i "meta\|canary\|diagnosis\|patch\|hook" /home/ray/train.log
```

### 4.2 元学习详细日志

每次运行的日志在 `/tmp/skyrl-logs/meta_run_<timestamp>/` 下：

```bash
# 找到最新的运行目录
ls -lt /tmp/skyrl-logs/meta_run_* | head -5

# 查看最新运行的周期摘要
cat /tmp/skyrl-logs/meta_run_<YYYYMMDD_HHMMSS>/cycle_summary.log

# 查看诊断器对话
cat /tmp/skyrl-logs/meta_run_<YYYYMMDD_HHMMSS>/diagnosis_conversations.jsonl | python3 -m json.tool

# 查看规划器对话
cat /tmp/skyrl-logs/meta_run_<YYYYMMDD_HHMMSS>/planning_conversations.jsonl | python3 -m json.tool
```

### 4.3 查看当前生效的补丁

```bash
# 查看当前 terminus-1 有哪些补丁
cd /home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus1
ls *.yaml hooks/*.py 2>/dev/null
git log --oneline

# 查看某个 hook 的内容
cat hooks/before_llm_call.py
```

### 4.4 Wandb

访问 Wandb 项目 `meta`，查看 `reward/avg_raw_reward` 曲线。元学习相关指标在 `meta_rl/` 前缀下。

## 5. 故障排查

### 5.1 奖励全为 0

**可能原因**：
- 智能体初始化失败（查看 `train.log` 中的 `HookExecutor` 或 `ImportError`）
- 补丁目录中有损坏的 hook 文件（清理后重试）
- 智能体名称和补丁目录不匹配

**排查步骤**：
```bash
grep -i "error\|exception\|traceback" /home/ray/train.log | head -20
```

### 5.2 金丝雀评估全部失败

**可能原因**：
- 上次运行残留的有害补丁（清理 meta_patches 目录）
- hook 代码有 bug（查看 `train.log` 中的 `Hook.*raised` 警告）

### 5.3 诊断器或规划器无输出

**可能原因**：
- LLM 服务不可用（检查 `META_LLM_BASE_URL`）
- token 限制太小（检查 `generator.meta.llm_max_tokens`）

**排查步骤**：
```bash
# 查看诊断器日志
ls -la /tmp/skyrl-logs/meta_run_*/diagnosis_conversations.jsonl
# 查看规划器日志
ls -la /tmp/skyrl-logs/meta_run_*/planning_conversations.jsonl
```

### 5.4 训练恢复后元学习状态不对

元学习的补丁状态保存在磁盘上的 `meta_patches/` 目录中，而不是 checkpoint 里。如果你从 checkpoint 恢复训练但清理了补丁目录，元学习会从零开始（这通常是期望的行为）。

## 6. 典型工作流

### 6.1 全新实验

```bash
# 1. 修改 run_megatron.sh 中的 RUN_NAME（用新名字）
# 2. 确认智能体和补丁目录匹配
# 3. 清理补丁目录
# 4. 启动训练
bash /home/ray/SkyRL/examples/train_integrations/harbor/run_megatron.sh 2>&1 | tee /home/ray/train.log
```

### 6.2 中断后继续

```bash
# 不需要清理补丁目录（保留已接受的补丁）
# 确保 RUN_NAME 不变，resume_mode=latest
bash /home/ray/SkyRL/examples/train_integrations/harbor/run_megatron.sh 2>&1 | tee -a /home/ray/train.log
```

### 6.3 切换智能体

```bash
# 1. 修改 run_megatron.sh 中的 agent.name 和 override_base
# 2. 换一个新的 RUN_NAME
# 3. 清理目标智能体的补丁目录
# 4. 启动训练
```
