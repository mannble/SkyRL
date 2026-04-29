def hook(command_keystrokes, terminal_output, context):
    return terminal_output + '\n[TIMEOUT] This command was too long or complex. Break it into smaller steps.'