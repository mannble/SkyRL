def hook(terminal_output, is_task_complete, context):
    rsync_warning = context.kv.pop('rsync_warning', False)
    if rsync_warning:
        return {
            'next_prompt': '[ACTION REQUIRED] You have run rsync multiple times. This may corrupt the log file by adding duplicate entries. Ensure rsync is run exactly once for the required task.',
            'prompt_mode': 'append'
        }
    return {}