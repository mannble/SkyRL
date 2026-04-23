def hook(commands, context):
    filtered = []
    for cmd in commands:
        if 'su - user -s /bin/bash -c' in cmd.keystrokes and 'crontab' in cmd.keystrokes:
            continue
        if cmd.keystrokes.strip():
            filtered.append(cmd)
    return filtered
