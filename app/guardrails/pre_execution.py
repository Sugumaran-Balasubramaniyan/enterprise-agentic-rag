import re
from typing import Tuple, Optional

class PreExecutionGuardrail:
    JAILBREAK_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
        r"you\s+are\s+now\s+(in\s+)?(developer|dan|god)\s+mode",
        r"system\s+prompt\s+override",
        r"reveal\s+(your|the)\s+(initial|system)\s+instructions",
        r"disregard\s+all\s+safety\s+filters"
    ]
    
    SQL_INJECTION_PATTERNS = [
        r"(\bUNION\b.*\bSELECT\b)",
        r"(\bDROP\b\s+\bTABLE\b)",
        r"(--|;|/\*|\*/)"
    ]

    @classmethod
    def inspect(cls, user_prompt: str) -> Tuple[bool, Optional[str]]:
        if not user_prompt or len(user_prompt.strip()) == 0:
            return False, "Empty query received."
            
        if len(user_prompt) > 4000:
            return False, "Input length exceeds security threshold (4000 chars)."

        for pattern in cls.JAILBREAK_PATTERNS:
            if re.search(pattern, user_prompt, re.IGNORECASE):
                return False, f"Adversarial prompt detected: matches security rule '{pattern}'"

        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, user_prompt, re.IGNORECASE):
                return False, "Malicious database syntax detected in query."

        return True, None
