def hook(terminal_output, context):
    # Limit terminal output length to prevent context overload
    if len(terminal_output) > 4000:
        # Keep first 1500 and last 1500 chars to preserve context
        terminal_output = terminal_output[:1500] + "\n... (output truncated) ...\n" + terminal_output[-1500:]
    return terminal_output
