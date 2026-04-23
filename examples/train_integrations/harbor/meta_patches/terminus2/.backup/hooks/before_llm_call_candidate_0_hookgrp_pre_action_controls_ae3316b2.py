def hook(prompt, context):
    context.kv.setdefault('total_llm_calls', 0)
    context.kv['total_llm_calls'] += 1
    if context.kv['total_llm_calls'] > 20:
        prompt += '\n\n[REMINDER] You have made many LLM calls. Stay focused on the original task and avoid repeating previous steps.'
    return prompt