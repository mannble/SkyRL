def hook(terminal_output, is_task_complete, context):
    # Prevent verification loops and context overload
    
    # Detect if agent is stuck in verification loop
    if context.kv.get("verify_count", 0) > 3:
        context.kv.setdefault("verify_count", 0)
        
        if not is_task_complete:
            # Inject guidance to reduce verification behavior
            return {
                "inject": (
                    "[GUIDANCE] You have run many verification commands. "
                    "Consider creating files in single commands instead of incrementally. "
                    "Run final verification once and mark task complete."
                ),
                "prompt_mode": "append"
            }
    
    # Handle task completion with verification loop prevention
    if is_task_complete:
        count = context.kv.setdefault("complete_count", 0)
        context.kv["complete_count"] = count + 1
        
        if count == 0:
            # First completion claim - ask for verification first
            return {
                "request_new_turn": True,
                "next_prompt": (
                    "[VERIFY BEFORE COMPLETE] You marked task_complete. "
                    "Please verify your work with ONE command (cat or ls) to confirm, "
                    "then submit task_complete again."
                ),
                "prompt_mode": "append"
            }
        
        # Second completion - allow actual completion
        context.kv["complete_count"] = 0
        return {}
    
    # Loop detection for repeated commands
    history = context.kv.setdefault("cmd_history", [])
    for cmd in context.last_commands:
        history.append(cmd.strip())
    if len(history) > 20:
        history[:] = history[-20:]
    
    if len(history) >= 4 and history[-2:] == history[-4:-2]:
        return {
            "inject": (
                "[LOOP DETECTED] You are repeating commands. "
                "Try a different approach or complete the task."
            ),
            "prompt_mode": "append"
        }
    
    return {}