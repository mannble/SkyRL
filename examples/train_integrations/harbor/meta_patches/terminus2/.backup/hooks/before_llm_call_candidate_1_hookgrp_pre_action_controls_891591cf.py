def hook(prompt, context):
    # Enforce compact JSON and prevent special characters in thought blocks
    instructions = '''IMPORTANT JSON FORMATTING RULES:
1. Output MUST be valid compact JSON (no extra whitespace, newlines, or indentation)
2. Example: {"analysis":"text","tool":"submit_diagnosis","diagnoses":[]}
3. NEVER include HTML entities like &emsp;, &nbsp;, etc.
4. NEVER include markdown code blocks (```json ... ```)
5. NEVER include trailing commas
6. Escape all special shell characters in command strings properly
7. Keep commands simple and testable - avoid complex heredocs with special chars
'''
    if 'JSON' not in prompt and 'json' not in prompt.lower():
        prompt = instructions + '\n\n' + prompt
    return prompt