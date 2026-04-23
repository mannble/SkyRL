def hook(terminal_output, is_task_complete, context):
    # Track verification commands to detect loops
    kv = context.kv
    
    # Initialize tracking structures
    cmd_history = kv.setdefault('cmd_history', [])
    verification_count = kv.setdefault('verify_count', 0)
    
    # Clean up history (keep last 20 commands)
    if len(cmd_history) > 20:
        cmd_history[:] = cmd_history[-20:]
    
    # Check for verification commands
    verify_cmds = ['cat', 'ls', 'head', 'tail', 'grep']
    is_verifying = any(cmd in context.last_commands for cmd in verify_cmds)
    
    # Detect verification loops
    if is_verifying:
        verification_count += 1
        # If we've verified the same thing 3+ times, suggest breaking the loop
        if verification_count >= 3:
            return {
                'request_new_turn': True,
                'next_prompt': (
                    '[LOOP DETECTED] You are repeatedly verifying the same files.\n'
                    'The task appears to be complete.\n'
                    'Please run a final comprehensive check and then mark complete.\n'
                    'If verification passes, confirm completion immediately.'
                ),
                'prompt_mode': 'append'
            }
    else:
        verification_count = 0
    
    # Track command history
    cmd_history.extend(context.last_commands)
    
    # Detect command parsing issues (incomplete redirects, malformed commands)
    for cmd in context.last_commands:
        if '>' in cmd and cmd.count('>') > 2:
            return {
                'request_new_turn': True,
                'next_prompt': (
                    '[COMMAND ERROR] Complex redirection detected.\n'
                    'Simplify your commands: use single redirections only.\n'
                    'Original task: ' + context.original_instruction[:150]
                ),
                'prompt_mode': 'append'
            }
    
    return {}
