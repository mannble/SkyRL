def hook(terminal_output, context):
    # Mitigate context_overload for complex multi-line commands (Task 944)
    # 1. Truncate excessive output that bloats context (e.g., large heredocs)
    if len(terminal_output) > 4000:
        terminal_output = terminal_output[:1500] + "\n... (output truncated for context) ...\n" + terminal_output[-2000:]
        context.kv.setdefault("context_savings", 0)
        context.kv["context_savings"] += len(terminal_output) - 4000
    # 2. Detect common command parsing failures (missing newlines, shell redirection issues)
    output_lines = terminal_output.strip().split('\n')
    if len(output_lines) > 10:  # Potential multi-line command output
        context.kv.setdefault("cmd_parse_flags", "")
        context.kv["cmd_parse_flags"] += "multi_line_output,"  # Flag for next LLM turn
    # 3. Track failed commands to avoid context loops
    if 'Permission denied' in terminal_output or 'command not found' in terminal_output:
        context.kv.setdefault("parse_errors", 0)
        context.kv["parse_errors"] += 1
        if context.kv["parse_errors"] > 2:
            context.kv.setdefault("loop_warning", 0)
            context.kv["loop_warning"] += 1
    return terminal_output