def hook(terminal_output, context):
    # Enforce strict heredoc closing
    if 'EOF' in terminal_output.upper() and 'EOF' not in context.kv.setdefault('last_heredoc_output', ''):
        context.kv['last_heredoc_output'] = terminal_output
        context.kv.setdefault('heredoc_check_count', 0)
    return terminal_output
