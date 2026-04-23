def hook(prompt, context):
    # Track call count to prevent excessive loops
    call_count = context.kv.setdefault("llm_call_count", 0)
    context.kv["llm_call_count"] = call_count + 1
    
    # Inject guidance for complex tasks to reduce context growth
    if call_count >= 5 and context.original_instruction:
        orig = context.original_instruction.strip()[:150]
        prompt += f"\n\n[CONTEXT CONSTRAINT] You are approaching context limits ({call_count} calls).\n"
        prompt += f"Task: {orig}\n"
        prompt += "- Use SIMPLE, SINGLE-LINE commands where possible\n"
        prompt += "- Avoid complex heredocs with many lines\n"
        prompt += "- If creating multi-line content, split into separate commands\n"
        prompt += "- Verify each step before proceeding\n"
    
    return prompt