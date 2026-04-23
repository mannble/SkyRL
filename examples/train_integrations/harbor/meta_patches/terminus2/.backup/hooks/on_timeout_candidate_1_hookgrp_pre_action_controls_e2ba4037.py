def hook(command_keystrokes, terminal_output, context):
    # Add timeout context to help agent understand command issues
    return terminal_output + "\n[TIMEOUT] Command timed out. The command may be too complex or missing proper syntax. Try breaking it into smaller steps."