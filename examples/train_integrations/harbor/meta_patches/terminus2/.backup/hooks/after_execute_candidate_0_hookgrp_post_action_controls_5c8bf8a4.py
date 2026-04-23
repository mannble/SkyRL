def hook(terminal_output, context):
    # Track all executed commands for verification logging
    commands_log = context.kv.setdefault('commands_log', [])
    
    # Extract commands from last_commands if available
    last_commands = getattr(context, 'last_commands', [])
    if last_commands:
        for cmd in last_commands:
            cmd_str = cmd.strip() if isinstance(cmd, str) else str(cmd)
            if cmd_str and not cmd_str.startswith(' ') and not cmd_str.startswith('\t'):
                commands_log.append(cmd_str)
    
    return terminal_output