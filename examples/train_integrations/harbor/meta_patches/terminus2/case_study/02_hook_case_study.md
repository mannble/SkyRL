# Hook 案例研究（37 个文件）

## 评估方法

- 源目录：`/home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus2/hooks`
- 评审维度：
  - 运行时安全性（是否可能抛错或破坏命令流）
  - 行为有效性（是否提升任务完成率）
  - 精确性（误报少、过度干预少）
  - 可维护性（逻辑清晰、重复度低）

## 补丁组合快照

- Hook 总数：`37`
- 按 Hook 点位分布：
  - `before_llm_call`：`8`
  - `before_execute`：`8`
  - `on_timeout`：`5`
  - `after_execute`：`8`
  - `after_round`：`8`
- 多数 Hook 使用 `context.kv` 维护状态，这提高了自适应能力，但也会放大跨轮次错误传播。

## 评分标准

- `A`：稳健且高价值
- `B`：有用，存在可控注意点
- `C`：脆弱/过拟合，建议重构
- `D`：明确 bug 或高风险

## 文件级评估

### before_llm_call

- `before_llm_call_s14_candidate_0_hookgrp_pre_action_controls_2e0d547d.py` -> `B`  
  对完成循环有一定纠偏作用，但存在过度提示风险。
- `before_llm_call_s10_candidate_0_hookgrp_pre_action_controls_1506fcd9.py` -> `C`  
  验证循环检测逻辑较弱，核心比较条件信息量不足。
- `before_llm_call_candidate_0_hookgrp_pre_action_controls_e8f92ae8.py` -> `B`  
  对重复 `cat/echo` 有一定抑制效果。
- `before_llm_call_candidate_1_hookgrp_pre_action_controls_e1888498.py` -> `B`  
  对 dpkg/checksum 场景的提醒较实用。
- `before_llm_call_candidate_1_hookgrp_pre_action_controls_d2d59c8d.py` -> `B`  
  基于上下文预算的提示总体有效且较安全。
- `before_llm_call_candidate_1_hookgrp_pre_action_controls_b871ce71.py` -> `C`  
  提示改写过强（JSON 约束），可能与解析器预期冲突。
- `before_llm_call_candidate_1_hookgrp_pre_action_controls_0bbc3050.py` -> `C`  
  验证计数逻辑存在不一致，计数更新位置可疑。
- `before_llm_call_candidate_1_hookgrp_pre_action_controls_3ddce9d2.py` -> `C`  
  偏元提醒，直接收益较低，且与其它 Hook 重复较多。

### before_execute

- `before_execute_s14_candidate_0_hookgrp_pre_action_controls_91619836.py` -> `B`  
  对空命令/畸形命令有基础过滤，属于合理护栏。
- `before_execute_s10_candidate_0_hookgrp_pre_action_controls_a846437a.py` -> `C`  
  去重有价值，但自动注入 `mkdir -p ... &&` 假设过强。
- `before_execute_candidate_1_hookgrp_pre_action_controls_77619f48.py` -> `C`  
  意图正确，但复杂命令分支触发条件偏弱。
- `before_execute_candidate_1_hookgrp_pre_action_controls_30f1f9d1.py` -> `D`  
  路径“归一化”可能把整条命令改写成无效绝对字符串。
- `before_execute_candidate_1_hookgrp_pre_action_controls_022ffa35.py` -> `D`  
  自动包裹 `su - user -c` 脆弱，易引发引号和执行语义问题。
- `before_execute_candidate_1_hookgrp_pre_action_controls_de3fac5a.py` -> `B`  
  轻量清洗，整体风险较低。
- `before_execute_candidate_1_hookgrp_pre_action_controls_c597eb6c.py` -> `C`  
  可工作，但包含转义告警模式，且启发式改写较宽泛。
- `before_execute_candidate_0_hookgrp_pre_action_controls_1c7ba974.py` -> `B`  
  简单的“表面检查”抑制逻辑，整体安全。

### on_timeout

- `on_timeout_s14_candidate_0_hookgrp_pre_action_controls_be501739.py` -> `B`  
  超时恢复提示清晰，风险低。
- `on_timeout_candidate_1_hookgrp_pre_action_controls_e2ba4037.py` -> `B`  
  简洁稳定。
- `on_timeout_candidate_1_hookgrp_pre_action_controls_274f5346.py` -> `B`  
  超时累计+升级提醒逻辑合理。
- `on_timeout_candidate_1_hookgrp_pre_action_controls_0672a5c8.py` -> `B`  
  长命令检测有价值，但有一定过拟合风险。
- `on_timeout_candidate_0_hookgrp_pre_action_controls_a92a297b.py` -> `C`  
  对 `printf` 的提示过于场景化，可能误导其它任务。

### after_execute

- `after_execute_candidate_1_hookgrp_post_action_controls_e8ed3dd8.py` -> `B`  
  输出截断与验证计数较实用。
- `after_execute_candidate_1_hookgrp_post_action_controls_e7ace1fd.py` -> `C`  
  向输出注入 ANSI 颜色转义序列，噪声较大且必要性低。
- `after_execute_candidate_1_hookgrp_post_action_controls_321b92b4.py` -> `B`  
  heredoc 失败追踪有帮助，风险低。
- `after_execute_candidate_1_hookgrp_post_action_controls_a23899ac.py` -> `C`  
  依赖精确列表匹配（`'write' in context.last_commands`），触发可靠性不足。
- `after_execute_candidate_1_hookgrp_post_action_controls_d5b1ecce.py` -> `C`  
  通过输出前缀识别验证循环，启发式偏弱。
- `after_execute_candidate_1_hookgrp_post_action_controls_082538f5.py` -> `C`  
  思路正确，但上下文节省量的账本计算有误。
- `after_execute_candidate_0_hookgrp_post_action_controls_d909fbbb.py` -> `C`  
  条件优先级问题可能导致无关输出被误判为“创建成功”。
- `after_execute_candidate_0_hookgrp_post_action_controls_4d7741ab.py` -> `B`  
  多信号解析/shebang 检测有价值，但建议降低启发式耦合。

### after_round

- `after_round_s14_candidate_0_hookgrp_post_action_controls_5943530c.py` -> `B`  
  目标正确（打断循环），但命令检测需更细致的子串匹配。
- `after_round_candidate_1_hookgrp_post_action_controls_3151307f.py` -> `B`  
  完成门控较保守，与质量目标一致。
- `after_round_candidate_0_hookgrp_post_action_controls_8d872b17.py` -> `B`  
  对 timeout/log-level 混淆场景有合理保护。
- `after_round_candidate_1_hookgrp_post_action_controls_29d55b1a.py` -> `C`  
  与其它验证循环控制重复较多，且可能引入噪声。
- `after_round_candidate_1_hookgrp_post_action_controls_b1215fca.py` -> `D`  
  使用了 `context.episode.original_instruction`（无效属性链），风险高。
- `after_round_candidate_0_hookgrp_post_action_controls_1fab62c7.py` -> `B`  
  逻辑较丰富，整体有用，但提示偏长。
- `after_round_candidate_1_hookgrp_post_action_controls_77441313.py` -> `B`  
  完成循环处理和重置语义较清晰。
- `after_round_candidate_0_hookgrp_post_action_controls_c4d0b384.py` -> `C`  
  存在多处弱条件（精确命令字符串匹配、不可达赋值）。

## 风险优先级观察（仅用于实验复盘）

- 高风险（`D`）优先关注：
  - `before_execute_candidate_1_hookgrp_pre_action_controls_30f1f9d1.py`
  - `before_execute_candidate_1_hookgrp_pre_action_controls_022ffa35.py`
  - `after_round_candidate_1_hookgrp_post_action_controls_b1215fca.py`
- 中风险（`C`）中建议重点复核：
  - `before_llm_call_s10_candidate_0_hookgrp_pre_action_controls_1506fcd9.py`
  - `before_llm_call_candidate_1_hookgrp_pre_action_controls_b871ce71.py`
  - `after_execute_candidate_1_hookgrp_post_action_controls_e7ace1fd.py`
  - `after_execute_candidate_1_hookgrp_post_action_controls_082538f5.py`
  - `after_execute_candidate_0_hookgrp_post_action_controls_d909fbbb.py`
  - `after_round_candidate_0_hookgrp_post_action_controls_c4d0b384.py`

## 总结

- Hook 覆盖面广、行为类型丰富，但质量分布不均。
- 当前版本可用于评测，但在长时训练中仍存在“问题 Hook 周期性污染”的潜在风险。
- 本文档结论用于实验记录与后续对比，不代表当前执行任何清理动作。
