def hook(commands, context):
    filtered = []
    for cmd in commands:
        if not cmd.keystrokes.strip():
            continue
        s = cmd.keystrokes.strip()
        if s.endswith('>') and '>' not in s.split('\n')[-2]:
            continue
        lines = s.split('\n')
        if len(lines) > 3:
            continue
        if 'EOF' in s or 'EOF\n' in s:
            context.kv.setdefault('heredoc_count', 0)
            context.kv['heredoc_count'] += 1
            if context.kv['heredoc_count'] >= 2:
                cmd.keystrokes = '# Use simple echo instead of heredoc' + cmd.keystrokes
        filtered.append(cmd)
    return filtered