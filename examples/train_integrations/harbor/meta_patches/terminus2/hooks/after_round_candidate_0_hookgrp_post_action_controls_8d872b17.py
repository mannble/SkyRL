def hook(terminal_output, is_task_complete, context):
    # Reset verification-specific state to prevent loops on unrelated tasks
    context.kv.setdefault('verification_attempts', 0)
    context.kv.setdefault('to_log_level_found', False)
    
    if is_task_complete:
        # Log completion attempt with context for verification analysis
        context.kv.setdefault('completion_attempts', 0)
        context.kv['completion_attempts'] += 1
        
        # Only add strict verification prompt if task explicitly mentions timeout/log_level
        if 'timeout' in context.original_instruction.lower() or 'log_level' in context.original_instruction.lower():
            return {
                'next_prompt': '[VERIFY] Confirming completion. Ensure all required outputs are correct.\n\nAre you sure you want to mark the task as complete?',
                'prompt_mode': 'append'
            }
    
    return {}