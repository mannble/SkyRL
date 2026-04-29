def hook(terminal_output, context):
    lower = terminal_output.lower()
    if "no such file" in lower or "not found" in lower:
        context.kv.setdefault("error_kind", "missing_path")
        context.kv.setdefault("last_failed_command", "")
        context.kv["last_failed_command"] = (context.last_commands[-1] if context.last_commands else "")
    elif "permission denied" in lower:
        context.kv.setdefault("error_kind", "permission")
        context.kv.setdefault("last_failed_command", "")
        context.kv["last_failed_command"] = (context.last_commands[-1] if context.last_commands else "")
    elif "json" in lower and ("parse error" in lower or "unexpected" in lower):
        context.kv.setdefault("error_kind", "json_parse")
        context.kv.setdefault("last_failed_command", "")
        context.kv["last_failed_command"] = (context.last_commands[-1] if context.last_commands else "")
    elif "zsh" in lower or "zsh: no such command" in lower:
        context.kv.setdefault("error_kind", "shell_syntax")
        context.kv.setdefault("last_failed_command", "")
        context.kv["last_failed_command"] = (context.last_commands[-1] if context.last_commands else "")
    context.kv.setdefault("last_output_chars", 0)
    context.kv["last_output_chars"] = len(terminal_output)