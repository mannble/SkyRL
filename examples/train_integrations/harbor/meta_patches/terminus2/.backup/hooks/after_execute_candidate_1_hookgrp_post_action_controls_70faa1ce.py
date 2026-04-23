def hook(terminal_output, context):
    if 'cat -A' in terminal_output.lower() or 'missing separator' in terminal_output.lower():
        if 'tab' not in terminal_output.lower():
            if context.kv.setdefault('makefile_attempt_count', 0) == 0:
                return terminal_output
        return terminal_output
    return terminal_output