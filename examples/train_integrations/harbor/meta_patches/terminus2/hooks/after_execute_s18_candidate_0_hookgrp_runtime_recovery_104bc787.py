def hook(terminal_output, context):
    # Check for missing files/directories
    lower = terminal_output.lower()
    if 'no such file' in lower or 'not found' in lower:
        cmd = context.last_commands[-1] if context.last_commands else ''
        context.kv['error_kind'] = 'missing_path'
        context.kv['last_failed_command'] = cmd[:200]
    elif 'permission denied' in lower:
        cmd = context.last_commands[-1] if context.last_commands else ''
        context.kv['error_kind'] = 'permission'
        context.kv['last_failed_command'] = cmd[:200]
    
    # Check for shell syntax errors (common in heredocs)
    if 'malformed here-document' in lower or 'unexpected end of file' in lower:
        context.kv['error_kind'] = 'shell_syntax'
    
    # Track output size for noise detection
    context.kv['last_output_chars'] = len(terminal_output)
