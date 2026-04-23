def hook(prompt, context):
    # Prevent context overload by enforcing concise structure
    context.kv.setdefault('verbose_call_count', 0)
    context.kv['verbose_call_count'] += 1
    
    # If we've already used many turns, inject strict JSON constraints
    if context.kv['verbose_call_count'] > 5:
        context.kv['verbose_call_count'] = 0  # Reset counter for new constraint
        prompt = prompt.replace('analysis:', '### ANALYSIS: ').replace('plan:', '### PLAN: ').replace('commands:', '### COMMANDS: ')
        prompt += '\n\n[CRITICAL CONSTRAINTS]\n1. OUTPUT VALID JSON ONLY (no markdown, no text before/after)\n2. Each command must be on a separate line (use \"cmd1\"\n\"cmd2\" in commands list)\n3. Keep analysis under 100 words\n4. Plan must be 1-2 sentences max\n5. Commands must not exceed 1500 chars total\n\n[PREVIOUS APPROACH FAILED]\nTask requires concise output. Avoid multi-line commands. Use printf instead of heredoc.'
    return prompt
