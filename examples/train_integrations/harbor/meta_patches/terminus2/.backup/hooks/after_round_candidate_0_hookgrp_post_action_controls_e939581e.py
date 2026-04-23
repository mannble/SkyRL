def hook(terminal_output, is_task_complete, context):
    # Enforce bidirectional config verification
    timeout_ok = 'timeout' in terminal_output.lower() or 'TO' in terminal_output.upper()
    log_level_ok = 'log_level' in terminal_output.lower() or 'loglevel' in terminal_output.lower() or 'LOG_LEVEL' in terminal_output.upper()
    
    if is_task_complete:
        context.kv['complete_verification_check'] = context.kv.setdefault('complete_verification_check', 0) + 1
        if context.kv['complete_verification_check'] > 1:
            return {}
        if not (timeout_ok and log_level_ok):
            return {
                'request_new_turn': True,
                'next_prompt': f'[VERIFY REQUIRED] Task is marked complete but verification failed.\nTerminal output: {terminal_output[:500]}\nPlease confirm BOTH changes are visible:\n1. timeout (TO)\n2. log_level (LOG_LEVEL)\nIf not found, apply changes again.',
                'prompt_mode': 'append'
            }
    return {}
