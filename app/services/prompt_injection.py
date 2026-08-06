# Deliberately basic (per spec: "basic prompt injection detection"): a
# keyword/phrase heuristic, not a trained classifier. This is a first line
# of defense - it catches unsophisticated, common injection phrasing, and
# will miss rephrased or more subtle attempts. Treat as one layer, not a
# guarantee.
_INJECTION_PATTERNS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard previous instructions",
    "disregard the system prompt",
    "you are now",
    "new instructions:",
    "system prompt:",
    "reveal your instructions",
    "act as if",
)


def detect_prompt_injection(text: str) -> bool:
    normalized = text.lower()
    return any(pattern in normalized for pattern in _INJECTION_PATTERNS)
