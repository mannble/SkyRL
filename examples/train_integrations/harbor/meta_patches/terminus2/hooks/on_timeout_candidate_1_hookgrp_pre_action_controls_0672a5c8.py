def hook(command_keystrokes, terminal_output, context):
    # Track timeout reasons
    timeout_count = context.kv.setdefault("timeout_count", 0)
    context.kv["timeout_count"] = timeout_count + 1
    
    # Check if timeout is due to long command
    if len(command_keystrokes) > 100:
        context.kv.setdefault("long_cmd_warning", "")
        context.kv["long_cmd_warning"] = context.kv.get("long_cmd_warning", "") + "\n"
        context.kv["long_cmd_warning"] += "[WARNING] Command was too long (>100 chars). Consider splitting into smaller commands.\n"
    
    return terminal_output