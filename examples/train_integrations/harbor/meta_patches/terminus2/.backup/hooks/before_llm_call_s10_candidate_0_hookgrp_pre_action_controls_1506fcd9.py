import re

def hook(prompt, context):
    # Prevent verification loops by tracking redundant commands
    kv = context.kv
    history = kv.setdefault('verify_history', [])
    last_3 = history[-3:]
    
    # Detect repeated cat/ls verification patterns
    for line in prompt.split('\n'):
        stripped = line.strip()
        if stripped.startswith('cat ') or stripped.startswith('ls '):
            history.append(stripped)
            if len(last_3) >= 3 and all(s in [last_3[0], last_3[1], last_3[2]] for s in [last_3[-1]]):
                if 'VERIFY' not in prompt.upper():
                    prompt += '\n[WARNING] Avoid repeated verification commands. Task is nearly complete.'
                break
    
    # Track call count for focused guidance
    kv.setdefault('call_count', 0)
    kv['call_count'] += 1
    if kv['call_count'] >= 10:
        prompt += '\n[FOCUS] Keep working toward task completion. Original task: ' + context.original_instruction[:150]
    
    return prompt