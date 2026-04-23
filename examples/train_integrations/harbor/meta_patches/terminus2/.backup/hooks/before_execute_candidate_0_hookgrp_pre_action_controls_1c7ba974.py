def hook(commands, context):
    filtered = []
    for cmd in commands:
        k = cmd.keystrokes.strip()
        if k.startswith('ls') or k.startswith('find') or k.startswith('dir'):
            context.kv.setdefault('superficial_check_count', 0)
            context.kv['superficial_check_count'] += 1
            if context.kv['superficial_check_count'] > 3:
                continue
        if k.startswith('cat') or k.startswith('head') or k.startswith('grep'):
            context.kv.setdefault('verification_count', 0)
            context.kv['verification_count'] += 1
        filtered.append(cmd)
    return filtered