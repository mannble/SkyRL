def hook(commands, context):
    # Filter out redundant verification commands
    seen_cmds = set()
    filtered = []
    
    for cmd in commands:
        stripped = cmd.keystrokes.strip() if hasattr(cmd, 'keystrokes') else str(cmd).strip()
        if stripped.startswith('cat ') or stripped.startswith('ls ') or stripped.startswith('head '):
            if stripped in seen_cmds:
                continue
            seen_cmds.add(stripped)
        filtered.append(cmd)
    
    # Ensure directories exist before find commands
    for cmd in filtered:
        keystrokes = cmd.keystrokes if hasattr(cmd, 'keystrokes') else str(cmd)
        if 'find ' in keystrokes and '-type f' not in keystrokes and '-perm' not in keystrokes:
            # Add mkdir before find if directory not mentioned
            if 'mkdir' not in keystrokes:
                cmd.keystrokes = 'mkdir -p /home/user/audit && ' + keystrokes
    
    # Cap duration for long-running commands
    for cmd in filtered:
        if hasattr(cmd, 'duration_sec'):
            cmd.duration_sec = min(cmd.duration_sec, 30)
    
    return filtered