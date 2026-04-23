def hook(terminal_output, context):
    # Track verification command patterns to detect loops
    kv = context.kv
    kv.setdefault('verification_cmds', [])
    
    # Clean output for analysis
    output = terminal_output.strip() if terminal_output else ''
    
    # Detect verification commands
    verif_cmds = ['cat ', 'ls ', 'wc -l', 'grep ', 'file ']
    detected = []
    for cmd in verif_cmds:
        if cmd in output:
            detected.append(cmd)
    
    # Track recent verification attempts (last 5 commands)
    if len(kv['verification_cmds']) > 5:
        kv['verification_cmds'] = kv['verification_cmds'][-5:]
    
    # If same verification commands repeated 3+ times without progress
    if detected and len(kv['verification_cmds']) >= 3:
        recent = kv['verification_cmds'][-3:]
        if detected in recent:
            # Inject guidance to break verification loop
            return terminal_output + '\n\n[GUIDANCE] You are over-verifying. \x1b[33mThe task appears complete. If verification passed, confirm completion directly.\x1b[0m'
    
    kv['verification_cmds'].extend(detected)
    return terminal_output
