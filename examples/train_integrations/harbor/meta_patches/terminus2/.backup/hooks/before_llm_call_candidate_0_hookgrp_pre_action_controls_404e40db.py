def hook(prompt, context):
    context.kv.setdefault('verification_level', 'basic')
    orig = context.original_instruction
    if 'report' in orig.lower() or 'log' in orig.lower():
        prompt += '\n\n[VERIFICATION REQUIRED] Before marking task complete:\n1. Verify all required files exist with cat or head\n2. Confirm file contents match expected format (check line endings, data structure)\n3. Do NOT complete until verification evidence is in terminal output'
    return prompt