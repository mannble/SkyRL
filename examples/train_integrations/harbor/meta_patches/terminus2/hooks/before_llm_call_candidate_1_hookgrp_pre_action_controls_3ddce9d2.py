def hook(prompt, context):
    # Track verification attempts to prevent infinite loops
    context.kv.setdefault('verification_attempts', 0)
    context.kv.setdefault('task_complete_attempts', 0)
    
    # Add reminder if we've tried to verify multiple times
    if context.kv['verification_attempts'] >= 2:
        prompt += "\n[DEBUG] You have verified this task multiple times. Ensure verification commands are complete and show full output."
    
    # Add reminder if task_complete has been attempted multiple times
    if context.kv['task_complete_attempts'] >= 2:
        prompt += "\n[DEBUG] You have attempted to mark task complete multiple times. Verify all requirements are met before confirming."
    
    return prompt