def hook(terminal_output, context):
    import re
    
    # Track file creation and check for missing shebangs
    if 'touch' in terminal_output or 'cat >' in terminal_output or 'echo' in terminal_output:
        if '.sh' in terminal_output or '.py' in terminal_output or '.txt' in terminal_output:
            context.kv.setdefault('file_create_count', 0)
            context.kv['file_create_count'] += 1
            
            # Check for missing shebang in shell scripts
            if '.sh' in terminal_output and 'echo' in terminal_output:
                if '#!/bin/bash' not in terminal_output and 'sudo ufw' in terminal_output:
                    # File likely created without shebang
                    context.kv.setdefault('shebang_missing', [])
                    context.kv['shebang_missing'].append(context.episode)
    
    # Detect malformed heredoc commands
    if 'EOF' in terminal_output:
        # Check for problematic patterns like '>> import' or extra redirections
        if re.search(r'>>\s*import|>>\s*from|>>\s*json|>>\s*os', terminal_output):
            context.kv.setdefault('heredoc_issues', [])
            context.kv['heredoc_issues'].append(context.episode)
            if len(context.kv['heredoc_issues']) <= 2:
                context.kv['heredoc_issues'].append('malformed_redirection')
    
    # Detect command parsing issues with special characters
    if '!' in terminal_output or '$' in terminal_output:
        if 'history expansion' in terminal_output.lower() or 'parsing error' in terminal_output.lower():
            context.kv.setdefault('parse_issues', 0)
            context.kv['parse_issues'] += 1
    
    # Track file sync issues
    if 'rsync' in terminal_output or 'scp' in terminal_output:
        context.kv.setdefault('sync_verified', False)
        context.kv['sync_verified'] = False
    
    return terminal_output