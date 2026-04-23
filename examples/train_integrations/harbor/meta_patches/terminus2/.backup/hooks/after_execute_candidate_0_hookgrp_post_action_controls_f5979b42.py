def hook(terminal_output, context):
    if len(terminal_output) > 5000:
        terminal_output = terminal_output[:2000] + '\n...truncated...\n' + terminal_output[-2000:]
    # Check for common command parsing issues in output
    if 'parsing' in terminal_output.lower() or 'error' in terminal_output.lower():
        context.kv.setdefault('parse_issues', 0)
        context.kv['parse_issues'] += 1
        terminal_output += '\n\n[DEBUG] Potential command parsing issue detected.'
    return terminal_output