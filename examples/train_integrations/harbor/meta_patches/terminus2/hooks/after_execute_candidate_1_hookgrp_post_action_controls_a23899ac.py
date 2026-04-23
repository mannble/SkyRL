def hook(terminal_output, context):
    import re
    current_log = context.kv.setdefault('last_log_content', '')
    
    # Check if we just created/modified a log file
    if 'write' in context.last_commands and 'log' in context.last_commands[-1].lower():
        # Update our record of what was written
        # Look for the echo/printf command output
        lines = terminal_output.split('\n')
        for line in lines:
            if line.strip().startswith('echo') or line.strip().startswith('printf'):
                current_log = line.strip().split('>>')[0] + current_log
        context.kv['last_log_content'] = current_log
    
    # Detect loops in log writing
    history = context.kv.setdefault('log_write_history', [])
    for cmd in context.last_commands:
        if 'echo' in cmd or 'printf' in cmd:
            history.append(cmd)
    if len(history) > 20:
        history = history[-20:]
    
    # Detect repeated log writes (loop detection)
    if len(history) >= 3 and history[-1] == history[-2] == history[-3]:
        return terminal_output + '\n\n[LOOP DETECTED] Stopping repeated log writes. Verify content manually.'
    
    return terminal_output