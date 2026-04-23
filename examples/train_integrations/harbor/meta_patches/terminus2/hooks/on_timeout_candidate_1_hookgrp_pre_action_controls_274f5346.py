def hook(command_keystrokes, terminal_output, context):
    # Add recovery guidance on timeout
    context.kv.setdefault('timeout_count', 0)
    context.kv['timeout_count'] += 1
    
    if context.kv['timeout_count'] > 1:
        return terminal_output + "\n[TIMEOUT WARNING] Multiple timeouts detected. Simplify commands and avoid long-running operations."
    
    return terminal_output