def hook(commands, context):
    context.kv.setdefault('user_switch_done', False)
    
    # If already switched to user account, skip
    if context.kv['user_switch_done']:
        return commands
    
    # Check if commands contain file operations (likely failing verification as root)
    file_ops = ['touch', 'echo', 'cat', 'printf', 'sed', 'awk', 'cp', 'mv', 'mkdir', 'ln']
    
    for i, cmd in enumerate(commands):
        if any(op in cmd.keystrokes for op in file_ops):
            # Prepend user switch command
            new_keystrokes = 'su - user -c "' + cmd.keystrokes + '"'
            cmd.keystrokes = new_keystrokes
            context.kv['user_switch_done'] = True
            break
    
    return commands