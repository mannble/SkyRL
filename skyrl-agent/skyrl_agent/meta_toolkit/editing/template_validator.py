"""Template validator: ensures modified agent templates won't break the runtime.

Validates placeholder presence, schema keywords, length bounds, and format
examples before allowing a template file to be written.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# Per-template validation rules
_TEMPLATE_RULES: dict[str, dict] = {
    "terminus-json-plain.txt": {
        "required_placeholders": ["{instruction}", "{terminal_state}"],
        "required_keywords": [
            '"analysis"', '"plan"', '"commands"', '"keystrokes"', '"duration"',
        ],
        "min_length": 300,
        "max_length": 15000,
        "description": "JSON system prompt template",
    },
    "terminus-xml-plain.txt": {
        "required_placeholders": ["{instruction}", "{terminal_state}"],
        "required_keywords": [
            "<analysis>", "<plan>", "<commands>", "<keystrokes",
        ],
        "min_length": 300,
        "max_length": 15000,
        "description": "XML system prompt template",
    },
    "timeout.txt": {
        "required_placeholders": ["{command}", "{timeout_sec}", "{terminal_state}"],
        "required_keywords": [],
        "min_length": 50,
        "max_length": 3000,
        "description": "Timeout feedback template",
    },
    "summarize-summary.txt": {
        "required_placeholders": ["{original_instruction}"],
        "required_keywords": [],
        "min_length": 100,
        "max_length": 5000,
        "description": "Summarization step 1 — summary generation prompt",
    },
    "summarize-questions.txt": {
        "required_placeholders": [
            "{original_instruction}", "{summary_content}", "{current_screen}",
        ],
        "required_keywords": [],
        "min_length": 100,
        "max_length": 5000,
        "description": "Summarization step 2 — question asking prompt",
    },
    "summarize-answers.txt": {
        "required_placeholders": ["{model_questions}"],
        "required_keywords": [],
        "min_length": 50,
        "max_length": 5000,
        "description": "Summarization step 3 — answer providing prompt",
    },
    "completion-confirmation-json.txt": {
        "required_placeholders": ["{terminal_output}"],
        "required_keywords": ['"task_complete"'],
        "min_length": 50,
        "max_length": 3000,
        "description": "Task completion confirmation (JSON format)",
    },
    "completion-confirmation-xml.txt": {
        "required_placeholders": ["{terminal_output}"],
        "required_keywords": ["<task_complete>"],
        "min_length": 50,
        "max_length": 3000,
        "description": "Task completion confirmation (XML format)",
    },
}


def validate_template(filename: str, content: str) -> ValidationResult:
    """Validate a template file's content against its rules.

    Args:
        filename: The template filename (e.g. "terminus-json-plain.txt")
        content: The proposed new content for the template

    Returns:
        ValidationResult with errors (blocking) and warnings (informational)
    """
    rules = _TEMPLATE_RULES.get(filename)
    if rules is None:
        return ValidationResult(
            valid=True,
            warnings=[f"No validation rules defined for '{filename}'; allowing write"],
        )

    errors: list[str] = []
    warnings: list[str] = []

    for ph in rules["required_placeholders"]:
        if ph not in content:
            errors.append(
                f"Missing required placeholder '{ph}' in {rules['description']}"
            )

    for kw in rules["required_keywords"]:
        if kw not in content:
            errors.append(
                f"Missing required schema keyword '{kw}' in {rules['description']}"
            )

    if len(content) < rules["min_length"]:
        errors.append(
            f"Template too short ({len(content)} chars < {rules['min_length']} min) "
            f"— likely truncated or incomplete"
        )

    if len(content) > rules["max_length"]:
        warnings.append(
            f"Template is very long ({len(content)} chars > {rules['max_length']} recommended) "
            f"— may waste context window"
        )

    # Check for accidentally doubled placeholders (common LLM mistake)
    for ph in rules["required_placeholders"]:
        count = content.count(ph)
        if count > 3:
            warnings.append(
                f"Placeholder '{ph}' appears {count} times — likely duplicated"
            )

    # Verify that str.format() can be called safely with only the declared
    # placeholders.  LLM-generated templates often contain JSON literals like
    # {"task_complete": true} whose braces are interpreted as format fields.
    placeholder_names = [
        ph.strip("{}") for ph in rules["required_placeholders"]
    ]
    dummy_kwargs = {name: "__TEST__" for name in placeholder_names}
    try:
        content.format(**dummy_kwargs)
    except KeyError as exc:
        errors.append(
            f"Template contains an unescaped brace that Python .format() "
            f"interprets as a placeholder: {exc}. Literal braces in JSON/text "
            f"must be doubled ({{{{ and }}}})."
        )
    except (ValueError, IndexError) as exc:
        errors.append(f"Template has malformed format string: {exc}")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


def get_known_templates() -> list[str]:
    """Return list of template filenames that have validation rules."""
    return list(_TEMPLATE_RULES.keys())


def get_template_rules(filename: str) -> dict | None:
    """Return the validation rules for a specific template, or None."""
    return _TEMPLATE_RULES.get(filename)
