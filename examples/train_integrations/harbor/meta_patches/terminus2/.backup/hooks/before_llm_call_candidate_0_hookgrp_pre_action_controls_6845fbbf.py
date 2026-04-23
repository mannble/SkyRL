def hook(prompt, context):
    # Prevent JSON parsing failures by reinforcing proper formatting
    context.kv.setdefault('json_format_warnings', 0)
    if context.kv['json_format_warnings'] >= 3:
        prompt += '\n\n[CRITICAL] Ensure all JSON is valid:\n1. All keys must be double-quoted: "key" not key\n2. Escape special chars: use \\\\n for newlines, \\\\\\\" for quotes\n3. No invalid escapes like \\4, \\x\n4. Each command must end with newline character\n5. Do not concatenate commands without newlines (cat filemkdir fails)'
    context.kv['json_format_warnings'] += 1
    return prompt