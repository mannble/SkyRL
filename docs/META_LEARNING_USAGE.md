# Meta-Learning 使用手册

## 1. 启动训练

```bash
bash /home/ray/SkyRL/examples/train_integrations/harbor/run_megatron.sh
```

训练日志输出到 `/home/ray/train.log`。

## 2. 查看运行状态

### 训练日志
```bash
tail -f /home/ray/train.log
```

### Meta 日志目录
每次运行会在 `/tmp/skyrl-logs/` 下创建带时间戳的目录：
```
/tmp/skyrl-logs/meta_run_YYYYMMDD_HHMMSS/
├── cycle_summary.log              # 每轮 meta 的人类可读总结
├── diagnosis_conversations.jsonl   # 诊断器 LLM 对话记录
├── planning_conversations.jsonl    # Planner LLM 对话记录
├── meta_conversations.jsonl        # 所有 LLM 对话（合并）
├── diagnosis_history.jsonl         # 跨 cycle 诊断历史
├── accepted_patches/               # 被接受的 patch JSON
└── harbor_default/
    └── meta_traces/
        └── traces.jsonl            # 轨迹记录
```

### 快速查看最新 cycle 结果
```bash
# 找到最新的 meta 运行目录
ls -td /tmp/skyrl-logs/meta_run_* | head -1

# 查看 cycle 总结
cat $(ls -td /tmp/skyrl-logs/meta_run_* | head -1)/cycle_summary.log
```

### 查看当前活跃补丁
```bash
ls /home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus2/
cat /home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus2/*.yaml 2>/dev/null
cat /home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus2/hooks/*.py 2>/dev/null
```

## 3. 停止训练

在 tmux 会话中按 `Ctrl+C`，或：
```bash
# 查找进程
ps aux | grep run_megatron
# 杀掉
kill <pid>
```

## 4. 清理残留文件（重新训练前必做）

### 4.1 清理补丁目录（最重要）

补丁目录使用独立 git 管理，被接受的 patch 会被 commit 进去。
**必须用 `git reset --hard` 回退到干净状态**，普通的 `git checkout` 不够。

```bash
cd /home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus2

# 找到干净的 commit（标记为 "Reset: clean all patches"）
git log --oneline | grep -i "reset\|clean\|initial"

# 回退到干净状态（用实际的 commit hash 替换）
git reset --hard 419d57c

# 验证：应该只剩 .gitignore 和 _templates
ls -la
```

或者用一行命令直接清理（删除所有 yaml 和 hooks，保留 .git 和 _templates）：
```bash
cd /home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus2
rm -f *.yaml
rm -rf hooks/
git add -A && git commit -m "Reset: clean all patches for fresh training run"
```

### 4.2 清理 meta 日志（可选）

```bash
rm -rf /tmp/skyrl-logs/meta_run_*
```

### 4.3 清理临时金丝雀目录（通常自动清理，以防万一）

```bash
rm -rf /tmp/meta_canary_*
```

## 5. 完整的重新训练流程

```bash
# 1. 停掉训练（tmux 中 Ctrl+C）

# 2. 清理补丁
cd /home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus2
rm -f *.yaml
rm -rf hooks/
git add -A && git commit -m "Reset: clean all patches for fresh training run"

# 3. 清理日志（可选）
rm -rf /tmp/skyrl-logs/meta_run_*
rm -rf /tmp/meta_canary_*

# 4. 重新启动
bash /home/ray/SkyRL/examples/train_integrations/harbor/run_megatron.sh
```

## 6. 关键配置参数

配置文件：`/home/ray/SkyRL/examples/train_integrations/harbor/run_megatron.sh`

| 参数 | 说明 | 当前值 |
|------|------|--------|
| `META_INTERVAL` | 每 N 个 batch 运行一次 meta | 1 |
| `META_CANDIDATES` | 每轮生成的候选补丁数 | 2 |
| `CANARY_TASKS` | 金丝雀评估任务数 | 8 |
| `CANARY_SAMPLES` | 每任务轨迹数 | 4 |
| `MAX_CONCURRENCY` | 最大并发数 | 32 |
| `META_LLM_MODEL` | Meta LLM 模型 | 见配置文件 |

## 7. 常见问题

### Q: 补丁没有被接受？
查看 `cycle_summary.log`，delta ≤ 0 的会被拒绝。这是正常的，说明金丝雀评估认为补丁没有带来改进。

### Q: 策略库没有产出？
确认 `cycle_summary.log` 中是否有 `strategy_library` 出现在 candidate 的 target_modules 中。
策略库需要 partial（半对半错）任务触发诊断器提取策略建议。

### Q: Hook 报错？
查看 `train.log` 中的 `Hook ... raised` 日志。Hook 报错不影响训练，会自动 fallback 到原始行为。

### Q: 金丝雀评估很慢？
金丝雀评估 = `CANARY_TASKS × CANARY_SAMPLES × (1 baseline + N candidates)` 个轨迹。
减小 `CANARY_TASKS` 或 `CANARY_SAMPLES` 可以加速。
