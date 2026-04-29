def hook(prompt, context):
    # Guard against multiple rsync runs causing log corruption
    rsync_count = context.kv.setdefault('rsync_count', 0)
    if rsync_count > 0:
        return {}
    
    # Initial reminder at episode 0 to identify deliverables and verify once
    if context.episode == 0:
        context.kv['rsync_count'] = 0
        return {
            'append_prompt': (
                '[REMINDER] Identify all required deliverable files and their exact formats before acting. '
                'After creating each file, verify its content matches the requirement. '
                'Avoid running creation commands multiple times unless explicitly required. '
                'Run verification checks once per deliverable, then confirm completion.'
            )
        }
    
    # Periodic reminder every 6-8 rounds to avoid repetition and reinforce single-run rule
    if context.episode > 0 and context.episode % 6 == 0:
        context.kv['reminders_sent'] = context.kv.setdefault('reminders_sent', 0) + 1
        if context.kv['reminders_sent'] < 3:  # Limit reminders
            return {
                'append_prompt': (
                    '[REMINDER] Review the task requirements. Ensure all deliverables are created '
                    'correctly and verified once. Do not repeat creation commands unnecessarily.'
                )
            }
    return {}