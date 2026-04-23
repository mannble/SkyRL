def hook(terminal_output, is_task_complete, context):
    kv = context.kv
    
    # Track completion attempts to prevent infinite loops
    kv.setdefault('completion_attempts', 0)
    kv.setdefault('verify_attempts', 0)
    
    if is_task_complete:
        kv['completion_attempts'] = kv.get('completion_attempts', 0) + 1
        
        # If we've tried to complete multiple times with verification issues, break the loop
        if kv.get('verify_attempts', 0) >= 2:
            # Reset verification counter to allow fresh attempt
            kv['verify_attempts'] = 0
            return {
                'request_new_turn': True,
                'next_prompt': '[LOOP DETECTED] You are stuck in a verification loop. You have marked task_complete multiple times. Please stop repeating verification commands and confirm completion directly if the task is done.',
                'prompt_mode': 'append'
            }
        
        # Track verification attempts
        kv['verify_attempts'] = kv.get('verify_attempts', 0) + 1
        
        # If complex commands were used, warn about potential parsing issues
        if kv.get('complex_cmd_count', 0) > 3:
            return {
                'inject': '[WARNING] You used multiple complex commands (variables, pipes). These may cause parsing failures. Simplify your next commands if verification fails.'
            }
    
    return {}
