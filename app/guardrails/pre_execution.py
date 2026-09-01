"""
Pre-execution guardrails for inspecting incoming user prompts before processing.
Detects prompt injections, obfuscated payloads (Base64, Hex, ROT13), SQL injections,
command injections, and RBAC policy violations.
"""

import base64
import codecs
import re
from typing import List, Optional, Tuple, Set


class PreExecutionGuardrail:
    """Hardened Pre-execution input validation and adversarial detection."""

    # Direct and indirect prompt injection patterns
    JAILBREAK_PATTERNS = [
        r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions",
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
        r"you\s+are\s+now\s+(in\s+)?(god|dan|developer|unrestricted|jailbreak)\s+mode",
        r"reveal\s+(your|the)\s+(initial|system)\s+instructions",
        r"reveal\s+(the\s+)?system\s+prompt",
        r"(show|print|output|display)\s+(the\s+)?(system\s+prompt|developer\s+prompt)",
        r"disregard\s+all\s+safety\s+filters",
        r"bypass\s+(all\s+)?(security|safety|content)\s+filters?",
        r"override\s+(all\s+)?(system|safety|security)\s+protocols?",
        r"act\s+as\s+an\s+unfiltered\s+ai",
        r"do\s+anything\s+now",
        r"system\s+prompt\s+override",
        r"\[SYSTEM\s+PROMPT\s+OVERRIDE\]",
    ]

    # SQL Injection detection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\bUNION\b.*\bSELECT\b)",
        r"(\bDROP\b\s+\b(TABLE|DATABASE|VIEW)\b)",
        r"(\bINSERT\b\s+\bINTO\b)",
        r"(\bDELETE\b\s+\bFROM\b)",
        r"(\b(OR|AND)\b\s+['\"]?1['\"]?\s*=\s*['\"]?1['\"])",
        r"(\b(OR|AND)\b\s+\d+\s*=\s*\d+)",
        r"(;\s*(DROP|SELECT|UPDATE|DELETE|INSERT|ALTER|EXEC|EXECUTE)\b)",
        r"(\b(SLEEP|BENCHMARK|WAITFOR\s+DELAY)\s*\()",
        r"(--|;|/\*|\*/)"
    ]

    # Shell / Command Injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r"\$\([^\)]+\)",
        r"`[^`]+`",
        r"(\|\s*(bash|sh|zsh|python|perl|eval|exec)\b)",
        r"(;\s*(rm\s+-rf|nc\s+-e|curl\s+.*\|\s*(bash|sh)|wget\s+.*\|\s*(bash|sh))\b)",
        r"(&&\s*(rm\s+-rf|cat\s+/etc/passwd|id|whoami)\b)",
    ]

    # Restricted keywords & topics mapped to required roles
    RESTRICTED_RESOURCES = [
        (r"(internal|system|database|root|admin)\s+(credentials?|passwords?|creds?)", "internal credentials"),
        (r"(private|secret|ssl|tls|ssh|pgp)\s+(encryption\s+)?keys?", "private encryption keys"),
        (r"(executive|ceo|cfo|cto|board)\s+(compensation|salaries|salary|bonuses?)", "executive compensation"),
        (r"(unredacted|confidential)\s+(m&a|merger|acquisition|financial\s+statements?)", "confidential M&A financials"),
    ]

    ADMIN_ROLES: Set[str] = {"admin", "administrator", "executive", "security_admin", "compliance_officer", "system_admin"}

    @classmethod
    def _check_obfuscated_payloads(cls, text: str) -> Optional[str]:
        """Detect and decode Base64, Hex, or ROT13 obfuscated malicious tokens."""
        # 1. Base64 detection
        b64_candidates = re.findall(r"[A-Za-z0-9+/]{8,}={0,2}", text)
        for cand in b64_candidates:
            try:
                # Pad to multiple of 4 if needed
                padded = cand + "=" * ((4 - len(cand) % 4) % 4)
                decoded_bytes = base64.b64decode(padded)
                decoded_text = decoded_bytes.decode("utf-8", errors="ignore").strip()
                if decoded_text:
                    for pattern in cls.JAILBREAK_PATTERNS + cls.COMMAND_INJECTION_PATTERNS:
                        if re.search(pattern, decoded_text, re.IGNORECASE):
                            return f"Obfuscated attack detected (Base64 decoded: '{decoded_text[:40]}...')"
            except Exception:
                continue

        # 2. Hex detection
        hex_candidates = re.findall(r"\b(?:0x)?[0-9a-fA-F]{16,}\b", text)
        for cand in hex_candidates:
            clean_hex = cand[2:] if cand.startswith("0x") else cand
            if len(clean_hex) % 2 == 0:
                try:
                    decoded_text = bytes.fromhex(clean_hex).decode("utf-8", errors="ignore").strip()
                    if decoded_text:
                        for pattern in cls.JAILBREAK_PATTERNS + cls.COMMAND_INJECTION_PATTERNS:
                            if re.search(pattern, decoded_text, re.IGNORECASE):
                                return f"Obfuscated attack detected (Hex decoded: '{decoded_text[:40]}...')"
                except Exception:
                    continue

        # 3. ROT13 detection
        try:
            rot13_text = codecs.decode(text, "rot_13")
            for pattern in cls.JAILBREAK_PATTERNS:
                if re.search(pattern, rot13_text, re.IGNORECASE) and not re.search(pattern, text, re.IGNORECASE):
                    return "Obfuscated attack detected (ROT13 encoded injection)"
        except Exception:
            pass

        return None

    @classmethod
    def inspect(cls, user_prompt: str, user_role: str = "standard_user") -> Tuple[bool, Optional[str]]:
        if not user_prompt or len(user_prompt.strip()) == 0:
            return False, "Empty query received."

        if len(user_prompt) > 4000:
            return False, "Input length exceeds security threshold (4000 chars)."

        for pattern in cls.JAILBREAK_PATTERNS:
            if re.search(pattern, user_prompt, re.IGNORECASE):
                return False, f"Adversarial prompt detected: matches security rule '{pattern}'"

        obfuscated_issue = cls._check_obfuscated_payloads(user_prompt)
        if obfuscated_issue:
            return False, obfuscated_issue

        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, user_prompt, re.IGNORECASE):
                return False, f"SQL injection pattern detected: {pattern}"

        for pattern in cls.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, user_prompt, re.IGNORECASE):
                return False, f"Shell/command injection pattern detected: {pattern}"

        normalized_role = (user_role or "standard_user").strip().lower()
        if normalized_role not in cls.ADMIN_ROLES:
            for pattern, resource_name in cls.RESTRICTED_RESOURCES:
                if re.search(pattern, user_prompt, re.IGNORECASE):
                    return False, f"Access denied: Role '{user_role}' is not authorized to request {resource_name}."

        return True, None


def validate_input(user_prompt: str, user_role: str = "standard_user") -> Tuple[bool, Optional[str]]:
    return PreExecutionGuardrail.inspect(user_prompt=user_prompt, user_role=user_role)

