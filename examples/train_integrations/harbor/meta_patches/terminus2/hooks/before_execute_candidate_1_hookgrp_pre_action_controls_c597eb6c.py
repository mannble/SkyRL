def hook(commands, context):
    # Filter out commands that might cause parsing failures
    valid_commands = []
    for cmd in commands:
        # Skip empty or malformed commands
        if not cmd.keystrokes or cmd.keystrokes.strip() == '':
            continue
        
        # Prevent shell variable expansion that breaks parsing
        keystrokes = cmd.keystrokes
        if '$(' in keystrokes or '\`' in keystrokes:
            # Replace shell expansion with literal strings or simpler commands
            if 'tar -tzf' in keystrokes:
                keystrokes = keystrokes.replace('$(', '(').replace(')', ')')
            elif 'ls -la' in keystrokes and ' ' in keystrokes:
                # Ensure proper spacing in ls commands
                pass
        
        # Cap duration to prevent timeouts from adding to context
        if cmd.duration_sec > 30:
            cmd.duration_sec = 30
        
        valid_commands.append(cmd)
    
    # Warn if commands are too numerous (potential context overload)
    if len(valid_commands) > 5:
        context.kv.setdefault('cmd_count', 0)
        context.kv['cmd_count'] += 1
        if context.kv['cmd_count'] > 10:
            # Inject warning to stop the loop
            context.kv.setdefault('stop_loop', True)
    
    return valid_commands
