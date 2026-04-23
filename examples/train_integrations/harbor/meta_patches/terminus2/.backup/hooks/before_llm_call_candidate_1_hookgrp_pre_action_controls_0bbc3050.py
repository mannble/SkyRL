def hook(prompt, context):
    context.kv.setdefault('json_call_count', 0)
    context.kv['json_call_count'] += 1
    
    # Prevent context overload from parsing loops
    if context.kv['json_call_count'] > 3:
        prompt += "\n[GUIDANCE] You must output valid JSON with exactly these fields: 'analysis', 'plan', 'commands'. Keep responses concise. Avoid verification loops."
    
    # Prevent excessive verification loops
    if 'Are you sure' in context.last_analysis or 'verify' in context.last_analysis.lower():
        if context.kv.get('verification_attempts', 0) > 2:
            prompt += "\n[WARNING] You have verified the same thing multiple times. Trust your work and mark task_complete if done."
            context.kv.setdefault('verification_attempts', 0)
            context.kv['verification_attempts'] += 1
    
    return prompt