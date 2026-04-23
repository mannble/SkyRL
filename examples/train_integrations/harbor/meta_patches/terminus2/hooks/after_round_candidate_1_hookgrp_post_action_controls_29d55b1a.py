def hook(terminal_output, is_task_complete, context):
    # Detect verification loops and inject refocusing guidance
    complete_count = context.kv.setdefault('complete_attempts', 0)
    if is_task_complete:
        complete_count += 1
        if complete_count > 1:
            return {
                'next_prompt': (
                    '[REFOCUS] You have marked task_complete multiple times. '
                    'Please review the original task requirements: ' +
                    str(context.episode.original_instruction)[:200] +
                    '\n\nBefore confirming completion, verify the exact format requirements are met.'
                ),
                'prompt_mode': 'append',
            }
    # Warn about potential verification confusion
    if 'timeout (TO)' in terminal_output or 'log_level (LOG_LEVEL)' in terminal_output:
        context.kv.setdefault('verification_confused', False)
        if not context.kv['verification_confused']:
            context.kv['verification_confused'] = True
    return {}