def hook(terminal_output, is_task_complete, context):
    # Track verification attempts to prevent infinite loops
    kv = context.kv
    kv.setdefault("verification_attempts", 0)
    
    # If task is marked complete, warn against immediate verification loops
    if is_task_complete:
        kv["verification_attempts"] += 1
        if kv["verification_attempts"] >= 2:
            return {
                "request_new_turn": True,
                "next_prompt": "[VERIFY] You marked task complete. Before confirming, please verify the required files exist and contain correct content. Do not run excessive verification commands in a loop. Confirm completion by stating 'task_complete' again only after verification is done."
            }
    else:
        # Reset counter if task not marked complete
        if not is_task_complete:
            kv["verification_attempts"] = 0
    
    return {}
