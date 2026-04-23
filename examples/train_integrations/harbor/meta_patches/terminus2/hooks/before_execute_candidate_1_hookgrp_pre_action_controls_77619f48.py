def hook(commands, context):
    # Limit verification commands to prevent over-verification
    verify_cmds = ['cat', 'ls', 'wc -l', 'grep', 'find']
    consecutive_verifies = 0
    filtered_commands = []
    
    for cmd in commands:
        if cmd.keystrokes.strip().startswith('cat ') or cmd.keystrokes.strip().startswith('ls ') or cmd.keystrokes.strip().startswith('wc') or cmd.keystrokes.strip().startswith('grep ') or cmd.keystrokes.strip().startswith('find '):
            consecutive_verifies += 1
            # Reject if 2+ consecutive verification commands
            if consecutive_verifies >= 2:
                continue
        else:
            consecutive_verifies = 0
        filtered_commands.append(cmd)
    
    # Reject overly complex multi-line commands that cause parsing failures
    for cmd in filtered_commands:
        if '\n' in cmd.keystrokes and ('heredoc' in context.last_analysis.lower() or 'EOF' in cmd.keystrokes):
            if context.kv.get('complex_cmd_count', 0) >= 2:
                context.kv.setdefault('complex_cmd_count', 0)
                context.kv['complex_cmd_count'] += 1
                continue  # Drop overly complex commands
    
    return filtered_commands