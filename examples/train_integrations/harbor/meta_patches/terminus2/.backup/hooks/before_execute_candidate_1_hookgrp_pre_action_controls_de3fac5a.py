def hook(commands, context):
    # Track execution attempts for this round
    exec_count = context.kv.setdefault("exec_count", 0)
    context.kv["exec_count"] = exec_count + 1
    
    # Filter empty or malformed commands
    filtered = []
    for cmd in commands:
        keystrokes = cmd.keystrokes.strip()
        if not keystrokes:
            continue
        
        # Prevent command concatenation issues - ensure proper newlines
        if '&&' in keystrokes or '||' in keystrokes:
            if exec_count >= 2:
                # Warn about complex chaining
                context.kv.setdefault("complex_cmd_warning", 0)
                context.kv["complex_cmd_warning"] += 1
        
        filtered.append(cmd)
    
    return filtered