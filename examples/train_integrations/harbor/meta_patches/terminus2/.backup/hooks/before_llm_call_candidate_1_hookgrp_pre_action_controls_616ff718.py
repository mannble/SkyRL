def hook(prompt, context):
    # Add efficiency warnings to prompt to prevent context overload
    call_count = context.kv.setdefault('llm_call_count', 0)
    context.kv['llm_call_count'] = call_count + 1
    
    # Track command usage patterns
    cmd_usage = context.kv.setdefault('cmd_usage', {})
    
    # Return the prompt as is, since no modification logic was provided
    return prompt