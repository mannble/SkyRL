def hook(terminal_output, context):
    import re
    lower = terminal_output.lower()
    if 'rsync' in lower:
        context.kv.setdefault('rsync_count', 0)
        context.kv['rsync_count'] += 1
        if context.kv['rsync_count'] > 1:
            context.kv['rsync_warning'] = True
        if 'already exists' in lower or 'overwrite' in lower:
            context.kv['last_rsync_output'] = terminal_output[:300]
            context.kv['rsync_warning'] = True
    return None