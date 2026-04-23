def hook(terminal_output, is_task_complete, context):
    # Prevent verification loops for completed tasks
    # Increment verification check counter
    kv = context.kv
    kv.setdefault('verify_count', 0)
    
    if is_task_complete:
        # If agent marked complete, check if we've already verified
        kv['verify_count'] = kv.get('verify_count', 0) + 1
        
        # If this is the first completion attempt, ask for verification
        if kv['verify_count'] == 1:
            return {
                'request_new_turn': True,
                'next_prompt': (
                    '[VERIFICATION REQUIRED] You marked the task complete.\n'
                    'Before confirming, please verify: The required output files exist and have the expected content.\n'
                    'If you are unsure, run verification commands first.'
                ),
                'prompt_mode': 'append'
            }
        else:
            # Second confirmation: allow completion
            return {
                'request_new_turn': False
            }
    
    # Reset counter if not complete
    kv['verify_count'] = 0
    return {}