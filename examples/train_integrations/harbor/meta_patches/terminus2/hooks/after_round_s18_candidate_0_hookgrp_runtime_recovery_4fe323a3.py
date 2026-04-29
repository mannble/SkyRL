def hook(terminal_output, is_task_complete, context):
    # Consume and act on error signals
    error_kind = context.kv.pop('error_kind', '')
    last_cmd = context.kv.pop('last_failed_command', '')
    if error_kind == 'missing_path':
        return {
            'next_prompt': (
                '[PATH ISSUE] A recent command referenced a missing path. '
                'List the parent directory or use find/ls before retrying. '
                + ('Command: ' + last_cmd if last_cmd else '')
            ),
            'prompt_mode': 'append',
        }
    if error_kind == 'permission':
        return {
            'next_prompt': (
                '[PERMISSION] A recent command hit a permission error. '
                'Check ownership/mode or choose a writable target before retrying. '
                + ('Command: ' + last_cmd if last_cmd else '')
            ),
            'prompt_mode': 'append',
        }
    if error_kind == 'shell_syntax':
        return {
            'next_prompt': (
                '[SHELL SYNTAX] A recent command had heredoc or quoting errors. '
                'Simplify quoting or use a plain heredoc before retrying.'
            ),
            'prompt_mode': 'append',
        }
    
    # Consume timeout signals
    timeout_kind = context.kv.pop('timeout_kind', '')
    if timeout_kind == 'broad_search':
        return {
            'next_prompt': (
                '[TIMEOUT] The previous command looked like an overly broad search. '
                'Narrow the path or pattern before retrying.'
            ),
            'prompt_mode': 'append',
        }
    if timeout_kind == 'install_or_network':
        return {
            'next_prompt': (
                '[TIMEOUT] Network or package installation timed out. '
                'Check connectivity or use a smaller install scope.'
            ),
            'prompt_mode': 'append',
        }
    
    # Consume completion signals
    complete_count = context.kv.setdefault('complete_count', 0)
    if is_task_complete:
        context.kv['complete_count'] = complete_count + 1
        if complete_count < 1:
            return {
                'next_prompt': (
                    '[COMPLETION CHECK] If the previous output already verifies '
                    'the required artifact, confirm completion now. Only run '
                    'another command if there is a concrete missing file, wrong '
                    'content, failed check, or unclear artifact location.'
                ),
                'prompt_mode': 'append',
            }
    else:
        context.kv['complete_count'] = 0
    
    return {}
