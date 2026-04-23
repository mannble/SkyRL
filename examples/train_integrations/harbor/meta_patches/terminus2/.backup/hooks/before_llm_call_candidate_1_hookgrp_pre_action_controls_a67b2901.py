def hook(prompt, context):
    context.kv.setdefault('json_guidance_injected', False)
    if not context.kv['json_guidance_injected']:
        context.kv['json_guidance_injected'] = True
        return prompt + '\n\n[JSON FORMATTING CRITICAL] When outputting reasoning or any JSON:\n1. ALL property names MUST use double quotes, e.g., {"analysis": "text"}'
        return prompt + '\n\n[JSON FORMATTING CRITICAL] When outputting reasoning or any JSON:\n1. ALL property names MUST be enclosed in double quotes (e.g., {"analysis": "text"}, NOT {analysis: "text"})\n2. No single quotes for keys\n3. Escape backslashes and quotes properly\n4. Keep JSON compact, avoid multi-line objects in single line'
    return prompt