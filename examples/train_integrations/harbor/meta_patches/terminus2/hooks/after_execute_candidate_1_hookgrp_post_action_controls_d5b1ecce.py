def hook(terminal_output, context):
    # Prevent context overload by truncating long outputs
    if len(terminal_output) > 3000:
        lines = terminal_output.split('\n')
        lines = lines[:1000] + ['... (output truncated) ...'] + lines[-1000:]
        terminal_output = '\n'.join(lines)
    
    # Track if we're running verification-only commands (cat, ls, head, tail)
    verification_cmds = ['cat ', 'ls ', 'head ', 'tail ', 'wc ', 'grep ']
    if any(cmd in terminal_output.split('\n')[0][:50] for cmd in verification_cmds):
        context.kv.setdefault("verify_count", 0)
        context.kv["verify_count"] += 1
    
    # Warn if too many verification commands in a row
    if context.kv.get("verify_count", 0) > 3:
        context.kv.setdefault("loop_warning", "")
        context.kv["loop_warning"] += "[WARNING] Running many verification commands. Consider completing task faster.\n"
    
    return terminal_output