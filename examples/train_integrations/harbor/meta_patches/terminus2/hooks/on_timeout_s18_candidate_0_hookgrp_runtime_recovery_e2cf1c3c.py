def hook(command_keystrokes, terminal_output, context):
    cmd = command_keystrokes.strip()
    context.kv['timeout_count'] = context.kv.setdefault('timeout_count', 0) + 1
    context.kv['last_timeout_command'] = cmd[:200]
    
    # Categorize timeout source
    if 'find /' in cmd or 'grep -R /' in cmd or 'grep -R .*' in cmd:
        context.kv['timeout_kind'] = 'broad_search'
    elif 'apt ' in cmd or 'pip ' in cmd or 'npm ' in cmd:
        context.kv['timeout_kind'] = 'install_or_network'
    elif 'tar ' in cmd or 'rsync ' in cmd:
        context.kv['timeout_kind'] = 'archive_transfer'
    else:
        context.kv['timeout_kind'] = 'long_command'
