def hook(prompt, context):
    context.kv.setdefault('verification_count', 0)
    context.kv.setdefault('file_creation_count', 0)
    
    # Warn about excessive verification loops
    if context.last_commands:
        cat_count = sum(1 for c in context.last_commands if 'cat ' in c and '/home/user/' in c)
        if cat_count >= 3:
            prompt += '\n[WARN] You are running multiple cat commands on the same file. Consider using a single comprehensive check instead of repeated verification.'
    
    # Warn about incremental file creation
    if context.last_commands:
        echo_count = sum(1 for c in context.last_commands if 'echo' in c or 'printf' in c)
        if echo_count >= 5:
            prompt += '\n[WARN] You are creating files with many separate echo/printf commands. Consider using a single printf with full content: printf \'line1\\nline2\\n\' > file.txt'
    
    return prompt
