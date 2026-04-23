def hook(terminal_output, is_task_complete, context):
    # Loop detection - track repeated command patterns
    cmd_hist = context.kv.setdefault('command_history', [])
    for cmd in context.last_commands:
        cmd_hist.append(cmd.strip())
    if len(cmd_hist) > 30:
        cmd_hist[:] = cmd_hist[-20:]
    
    # Detect verification loops (same file checked 3+ times)
    verif_count = context.kv.setdefault('verif_count', {})
    for cmd in context.last_commands:
        if cmd in ['cat', 'ls -la', 'wc -l', 'head', 'tail']:
            # Check if file already verified in this round
            if any(f'cat ' in c or 'ls -la' in c for c in context.last_commands[-3:]):
                verif_count['round'] = verif_count.get('round', 0) + 1
                if verif_count['round'] >= 3:
                    return {
                        "request_new_turn": True,
                        "next_prompt": ("[LOOP DETECTED] You are repeatedly verifying the same file. "
                        "The file has been checked multiple times this round. "
                        "Either the file is correct, or try a different approach. "
                        "Task: " + context.original_instruction[:150]),
                        "prompt_mode": "append",
                    }
                verif_count['round'] = 0
    
    # Detect stuck state from terminal output
    stuck_count = context.kv.get('stuck_count', 0)
    if stuck_count >= 3:
        return {
            "request_new_turn": True,
            "next_prompt": ("[STUCK STATE] The terminal appears to be in an unusual state. "
            "Try simplifying your next command. Avoid complex multi-line inputs. "
            "Task: " + context.original_instruction[:150]),
            "prompt_mode": "append",
        }
    
    # Context management - warn if approaching limits
    episode = context.episode
    total = context.total_episodes
    if episode >= total - 2:
        context.kv.setdefault('near_limit', True)
        if not context.kv.get('near_limit_warned'):
            return {
                "next_prompt": ("[SPEED WARNING] You are on episode " + str(episode) + "/" + str(total) + ". "
                "Focus on completing the task efficiently. Avoid unnecessary verification steps. "
                "Task: " + context.original_instruction[:150]),
                "prompt_mode": "append",
            }
            context.kv['near_limit_warned'] = True
    
    # Handle task completion verification
    if is_task_complete:
        context.kv.setdefault('completion_attempts', 0)
        context.kv['completion_attempts'] += 1
        if context.kv['completion_attempts'] > 1:
            return {}
        return {
            "next_prompt": ("[TASK COMPLETE] Before marking complete, verify one final time: "
            "1. Check all required files exist\n" +
            "2. Verify file contents match requirements\n" +
            "3. Ensure no errors in terminal output\n" +
            "If all checks pass, submit task_complete again."),
            "prompt_mode": "append",
        }
    
    # Command parsing failure detection
    if 'syntax' in terminal_output.lower() or 'error' in terminal_output.lower() or 'command' in terminal_output.lower():
        context.kv.setdefault('parse_errors', 0)
        context.kv['parse_errors'] += 1
        if context.kv['parse_errors'] >= 2:
            return {
                "next_prompt": ("[PARSING ISSUES] You are getting command parsing errors. "
                "Try: 1. Use simpler commands 2. Avoid complex quoting 3. Use single-line commands 4. Verify file paths\n" +
                "Task: " + context.original_instruction[:150]),
                "prompt_mode": "append",
            }
    
    return {}