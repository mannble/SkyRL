import re

def hook(commands, context):
    for cmd in commands:
        if cmd.keystrokes.strip().startswith('cat >') and '<<' in cmd.keystrokes:
            lines = cmd.keystrokes.split('\n')
            if lines[0].startswith('cat >'):
                lines[0] = 'cat > ' + lines[0].split('cat > ')[1]
            cmd.keystrokes = '\n'.join(lines)
        if cmd.duration_sec > 30:
            cmd.duration_sec = 10
    return [c for c in commands if c.keystrokes.strip()]