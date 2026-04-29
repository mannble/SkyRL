def hook(terminal_output, is_task_complete, context):
    timeout_kind = context.kv.pop("timeout_kind", "")
    if timeout_kind == "broad_search":
        return {
            "next_prompt": ("[TIMEOUT] The previous command looked like an overly broad search. " "Narrow the path or pattern before retrying."),
            "prompt_mode": "append",
        }
    error_kind = context.kv.pop("error_kind", "")
    cmd = context.kv.pop("last_failed_command", "") if error_kind else ""
    if error_kind == "missing_path":
        return {
            "next_prompt": ("[PATH ISSUE] A recent command referenced a missing path. " "List the parent directory or use find/ls before retrying. " + ("Command: " + cmd if cmd else "")),
            "prompt_mode": "append",
        }
    if error_kind == "permission":
        return {
            "next_prompt": ("[PERMISSION] A recent command hit a permission error. " "Check ownership/mode or choose a writable target before retrying. " + ("Command: " + cmd if cmd else "")),
            "prompt_mode": "append",
        }
    if error_kind == "json_parse":
        return {
            "next_prompt": ("[JSON ERROR] A recent command produced JSON parsing errors. " "Simplify the command, avoid complex quoting, or split it into smaller steps."),
            "prompt_mode": "append",
        }
    if error_kind == "shell_syntax":
        return {
            "next_prompt": ("[SHELL ERROR] A recent command had shell syntax issues. " "Simplify quoting, use a plain heredoc, or split the command."),
            "prompt_mode": "append",
        }
    timeout_count = context.kv.pop("timeout_count", 0)
    if timeout_count >= 2:
        return {
            "next_prompt": ("[TIMEOUT] Commands are timing out. Try a simpler, faster command or narrower search."),
            "prompt_mode": "append",
        }
    return {}