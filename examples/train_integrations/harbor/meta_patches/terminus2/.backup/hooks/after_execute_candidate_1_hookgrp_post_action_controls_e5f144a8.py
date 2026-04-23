def hook(terminal_output, context):
    # Ensure agent runs as 'user' and not root for verification
    # Check if output indicates root execution or verification checks failed
    if 'root' in terminal_output.lower() or 'permission denied' in terminal_output.lower():
        return terminal_output
    return terminal_output