def hook(terminal_output, context):
    # Track heredoc/command failures to warn agent
    if 'heredoc' in terminal_output.lower() or 'EOF' in terminal_output:
        context.kv.setdefault('heredoc_attempts', 0)
        context.kv['heredoc_attempts'] += 1
    # Warn if heredoc failed (non-empty output with error)
    if context.kv.get('heredoc_attempts', 0) > 2:
        context.kv.setdefault('heredoc_warned', False)
        if not context.kv['heredoc_warned']:
            context.kv['heredoc_warned'] = True
            terminal_output += "\n[WARN] Heredoc commands may fail in this environment. Consider using simple echo/printf instead."
    return terminal_output