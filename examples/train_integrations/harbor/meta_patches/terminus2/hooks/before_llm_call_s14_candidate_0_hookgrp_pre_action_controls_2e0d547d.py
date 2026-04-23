def hook(prompt, context):
    context.kv.setdefault('completion_attempts', 0)
    context.kv.setdefault('verify_count', 0)
    if context.is_task_complete:
        context.kv['verify_count'] += 1
        if context.kv['verify_count'] >= 3:
            prompt += '\n\n[WARNING] You have verified 3 times already. Stop verifying and confirm completion if task is done.'
    if context.kv['completion_attempts'] >= 2:
        prompt += f'\n\n[RESTART] Task incomplete after {context.kv["completion_attempts"]} attempts. Re-read the original instruction:\n{context.original_instruction[:500]}'
    return prompt