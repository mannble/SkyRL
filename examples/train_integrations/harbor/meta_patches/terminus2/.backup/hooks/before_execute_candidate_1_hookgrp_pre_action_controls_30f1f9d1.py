def hook(commands, context):
    # Normalize all paths to absolute to prevent path resolution failures
    for cmd in commands:
        keystrokes = cmd.keystrokes
        # Convert relative paths starting with './' or '../' to absolute
        if keystrokes:
            lines = keystrokes.split('\n')
            new_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith('#') and not stripped.startswith('$'):
                    # Check for relative path usage
                    if stripped.startswith('./') or stripped.startswith('../') or (stripped.startswith('/') == False and not stripped[0].isalpha() and not stripped[0] == '~'):
                        # Convert to absolute path if it looks like a path
                        # Simple heuristic: if it contains / and doesn't start with /
                        if '/' in stripped and not stripped.startswith('/'):
                            new_line = '/home/user' + stripped
                        elif stripped.startswith('./'):
                            new_line = stripped[2:] if stripped[2:] else ''
                        elif stripped.startswith('../'):
                            new_line = stripped[3:] if stripped[3:] else ''
                        else:
                            new_line = stripped
                    else:
                        new_line = stripped
                    new_lines.append(new_line)
                else:
                    new_lines.append(line)
            cmd.keystrokes = '\n'.join(new_lines)
    return commands