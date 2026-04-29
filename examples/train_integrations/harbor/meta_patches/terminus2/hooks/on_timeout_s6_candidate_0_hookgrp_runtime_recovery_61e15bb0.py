def hook(command_keystrokes, terminal_output, context):
    cmd = command_keystrokes.strip()
    context.kv.setdefault("timeout_count", 0)
    context.kv["timeout_count"] += 1
    context.kv.setdefault("last_timeout_command", "")
    context.kv["last_timeout_command"] = cmd[:200]
    if "find /" in cmd or "ls -R" in cmd:
        context.kv.setdefault("timeout_kind", "broad_search")
    elif "apt " in cmd or "pip " in cmd:
        context.kv.setdefault("timeout_kind", "install_or_network")
    else:
        context.kv.setdefault("timeout_kind", "long_command")