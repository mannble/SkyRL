def hook(terminal_output, context):
    # Fix cron command parsing: extract the raw crontab line from su output
    # Pattern: crontab -l output appears after su - user -s /bin/bash -c "echo CRON_ENTRY"
    import re
    cron_line = None
    
    # Detect if we just ran a cron add command with su
    if 'su' in terminal_output and ('CRON_ENTRY' in terminal_output or 'cron' in terminal_output.lower()):
        # Look for the actual crontab line that was added
        lines = terminal_output.split('\n')
        for line in lines:
            # Match lines that look like cron entries (e.g., "30 6 * * * /path/to/script")
            if re.match(r'^\d{1,2}.*\*', line) or 'CRON_ENTRY' in line:
                cron_line = line.strip()
                break
    
    # If we found a cron entry, append a verification command to the next prompt
    if cron_line and 'crontab -l' not in terminal_output:
        return terminal_output + "\n\n[VERIFY] The above cron entry was added. To verify, run: crontab -l"
    
    return terminal_output