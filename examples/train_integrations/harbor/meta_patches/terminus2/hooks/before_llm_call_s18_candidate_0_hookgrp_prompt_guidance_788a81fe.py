def hook(prompt, context):
    # Track if initial verification reminder sent
    if context.episode == 0:
        context.kv.setdefault('initial_verification_reminder', False)
        if not context.kv['initial_verification_reminder']:
            context.kv['initial_verification_reminder'] = True
            context.kv['verification_reminder_count'] = 0
            return {
                'append_prompt': (
                    '[VERIFICATION REMINDER] Before marking task complete, verify all deliverables match requirements exactly. ' 
                    'Check file contents with cat/cat -A, verify exact format, confirm file exists at correct path. ' 
                    'Do NOT mark complete without explicit verification of required output.'
                )
            }
    
    # Periodic reminder every 6 rounds for verification discipline
    if context.episode > 0 and context.episode % 6 == 0:
        count = context.kv.setdefault('verification_reminder_count', 0)
        if count < 2:
            context.kv['verification_reminder_count'] = count + 1
            return {
                'append_prompt': (
                    '[VERIFICATION REMINDER] Verify deliverables match exact requirements before completion. ' 
                    'Check for required fields, correct format, exact content, proper location. ' 
                    'Incomplete verification is a common failure pattern.'
                )
            }
    
    return {}