def hook(commands, context):
    cleaned = []
    for cmd in commands:
        # Remove HTML entities that break command parsing
        if '&emsp;' in cmd.keystrokes or '&nbsp;' in cmd.keystrokes:
            continue
        # Remove markdown code blocks
        if cmd.keystrokes.startswith('```') or cmd.keystrokes.endswith('```'):
            continue
        # Clean up command format
        keystrokes = cmd.keystrokes.strip()
        if keystrokes and not keystrokes.startswith('#'):
            # Ensure single command per entry, no multi-line confusion
            if '\n' in keystrokes and not keystrokes.startswith('echo'):
                # Skip complex multi-line commands that cause parsing issues
                continue
            cleaned.append(cmd)
    return cleaned