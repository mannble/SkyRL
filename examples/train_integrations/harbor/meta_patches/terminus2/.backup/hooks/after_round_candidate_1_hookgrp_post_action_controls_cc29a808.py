def hook(terminal_output, is_task_complete, context):
    task = context.original_instruction[:100].lower()
    needs_strict_verification = any(x in task for x in ['makefile', 'log file', 'config', 'hardening', 'report', 'restore', 'inodes', 'capacity', 'crontab', 'security'])
    if needs_strict_verification and is_task_complete:
        if context.kv.setdefault('complete_count', 0) < 2:
            return {
                'request_new_turn': True,
                'next_prompt': f'[VERIFY] Before marking complete, ensure all requirements are met. Original task: {context.original_instruction[:200]}\n1. Check file contents match exact specifications\n2. Verify line counts, formats, and permissions\n3. Confirm no trailing whitespace or incorrect tabs',
                'prompt_mode': 'append'
            }
    return {}