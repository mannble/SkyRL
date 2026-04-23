def hook(prompt, context):
    if 'Critical' in prompt and 'High' in prompt:
        if 'analysis' not in prompt.lower() and 'plan' not in prompt.lower():
            return prompt + '\n[STRUCTURE] Ensure JSON output is valid and includes: analysis, plan, commands fields.'
    return prompt
