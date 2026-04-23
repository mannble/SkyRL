def hook(terminal_output, context):
    # Track verification command loops
    verif_cmds = ['cat', 'ls', 'wc', 'head', 'tail', 'grep', 'find', 'diff', 'cmp']
    recent = context.kv.setdefault('recent_cmds', [])
    recent.append(terminal_output[:200])
    if len(recent) > 10:
        recent = recent[-5:]
    
    # Detect stuck states (multi-line prompts or truncated output)
    if '>' in terminal_output or terminal_output.endswith('...') or len(terminal_output) < 10 and terminal_output:
        context.kv.setdefault('stuck_count', 0)
        context.kv['stuck_count'] += 1
        if context.kv['stuck_count'] >= 2:
            return terminal_output + '\n[WARNING] Terminal appears stuck. Consider simplifying next command.'
        context.kv['stuck_count'] = 0
    
    # Track file creation attempts
    file_creators = ['touch', 'echo', 'printf', 'cat >', 'cat <<', 'tee', 'mv', 'cp', 'mkdir']
    for creator in file_creators:
        if creator in terminal_output and 'Created' in terminal_output or 'success' in terminal_output.lower():
            context.kv.setdefault('files_created_this_round', [])
            context.kv['files_created_this_round'].append(terminal_output[:100])
            if len(context.kv['files_created_this_round']) > 5:
                context.kv['files_created_this_round'] = context.kv['files_created_this_round'][-3:]
    
    # Detect repeated verification on same file
    if any('error' in terminal_output.lower() or 'fail' in terminal_output.lower() or 'incorrect' in terminal_output.lower() for line in terminal_output.split('\n') if line.strip()):
        context.kv.setdefault('last_error_file', None)
        for line in terminal_output.split('\n'):
            if 'File' in line or 'error' in line.lower():
                context.kv['last_error_file'] = line[:80]
    
    # Clean up excessive output
    if len(terminal_output) > 3000:
        parts = terminal_output.split('\n')
        if len(parts) > 50:
            context.kv.setdefault('output_truncated', True)
            return terminal_output[:1500] + '\n\n[OUTPUT TRUNCATED]' + '\n'.join(parts[-40:])
    return terminal_output