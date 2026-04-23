def hook(prompt, context):
    context.kv.setdefault('dpkg_redirect_count', 0)
    if context.episode == 0 and 'dpkg' in context.original_instruction.lower():
        prompt += '\n\n[IMPORTANT] When running dpkg commands, ALWAYS redirect output to a file using > or >>. Example: dpkg -l bash | grep ... > /path/to/log'
    if context.episode == 0 and 'checksum' in context.original_instruction.lower():
        prompt += '\n\n[IMPORTANT] When creating checksum logs, use exactly TWO spaces between hash and filename. Format: <hash>  <filename>'
    return prompt