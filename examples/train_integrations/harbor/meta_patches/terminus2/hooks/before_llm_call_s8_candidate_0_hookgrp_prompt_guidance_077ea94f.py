def hook(prompt, context):
    # Task discipline: identify deliverables and verification criteria early
    if context.episode == 0:
        context.kv.setdefault("reminders_sent", 0)
        context.kv["reminders_sent"] += 1
        return {
            "append_prompt": (
                "[TASK PLAN] Before running commands, identify all required output files, their exact content/format, and byte-level verification criteria. "
                "Plan to create files atomically (single heredoc/printf) and verify exact format (lines, bytes, no trailing blanks)."
            )
        }
    # Periodic reminder every 6 rounds to maintain focus
    if context.episode > 0 and context.episode % 6 == 0:
        return {
            "append_prompt": (
                "[REMINDER] Verify exact file names, formats, and byte counts against requirements before confirming completion."
            )
        }
    return {}
