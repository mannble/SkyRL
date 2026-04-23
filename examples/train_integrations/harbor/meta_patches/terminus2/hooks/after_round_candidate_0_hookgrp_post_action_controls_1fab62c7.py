def hook(terminal_output, is_task_complete, context):
    # Shebang verification - inject prompt if missing
    if context.kv.get('shebang_missing') and len(context.kv['shebang_missing']) > 1:
        if context.episode not in context.kv['shebang_missing']:
            context.kv['shebang_missing'].append(context.episode)
        return {
            'next_prompt': (
                '\n\n[CRITICAL] VERIFICATION FAILURE DETECTED\n'
                'Your script files are missing the required shebang line (#!/bin/bash)\n'
                'Before submitting, ensure every .sh file starts with:\n'
                '#!/bin/bash\n\n[Then your script commands]'
            ),
            'prompt_mode': 'append',
        }
    
    # Heredoc command issues detection
    if context.kv.get('heredoc_issues') and 'malformed_redirection' in context.kv['heredoc_issues'][-1]:
        return {
            'next_prompt': (
                '\n\n[COMMAND STRUCTURE ISSUE]\n'
                'Your heredoc commands contain invalid redirection operators\n'
                'Avoid patterns like: cat > file << EOF > import\n'
                'Use simple heredoc: cat > file << EOF\n'
                'EOF\n'
                'Then content\n'
                'EOF'
            ),
            'prompt_mode': 'append',
        }
    
    # Parse error tracking
    if context.kv.get('parse_issues', 0) > 3:
        return {
            'next_prompt': (
                '\n\n[PARSE ERROR FREQUENCY HIGH]\n'
                'You have had multiple command parsing failures.\n'
                'Simplify your commands: avoid shell variables, pipes, and special chars\n'
                'Use basic commands first, then build complexity'
            ),
            'prompt_mode': 'append',
        }
    
    # Context overload prevention
    if context.n_commands_executed > 15 and context.episode > 5:
        return {
            'next_prompt': (
                '\n\n[CONTEXT WARNING]\n'
                'You have executed many commands. Consider:\n'
                '1. Combining related operations\n'
                '2. Verifying each step before proceeding\n'
                '3. Breaking complex tasks into smaller chunks'
            ),
            'prompt_mode': 'append',
        }
    
    # Task completion guidance
    if is_task_complete:
        context.kv.setdefault('complete_count', 0)
        context.kv['complete_count'] += 1
        if context.kv['complete_count'] == 1:
            return {
                'request_new_turn': True,
                'next_prompt': (
                    '\n\n[TASK COMPLETION VERIFICATION]\n'
                    'You marked task_complete. Before confirming:\n'
                    '1. Verify all required files exist (ls -la)\n'
                    '2. Check file contents match requirements (cat filename)\n'
                    '3. Run verification commands if specified\n'
                    '4. Confirm shebang lines in all scripts'
                ),
                'prompt_mode': 'append',
            }
    
    return {}