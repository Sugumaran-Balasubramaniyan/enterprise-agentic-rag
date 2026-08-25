import re
from typing import Dict, Any, List, Tuple

class PostExecutionGuardrail:
    EMAIL_REGEX = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    CREDIT_CARD_REGEX = r"\b(?:\d[ -]*?){13,16}\b"

    @classmethod
    def sanitize_pii(cls, text: str) -> str:
        text = re.sub(cls.EMAIL_REGEX, "[REDACTED_EMAIL]", text)
        text = re.sub(cls.CREDIT_CARD_REGEX, "[REDACTED_CARD]", text)
        return text

    @classmethod
    def verify_factual_grounding(cls, answer: str, context_chunks: List[Dict[str, Any]]) -> Tuple[bool, float]:
        if not context_chunks:
            return True, 1.0
            
        answer_words = set(re.findall(r"\w+", answer.lower()))
        if not answer_words:
            return True, 1.0
            
        combined_context = " ".join([c["content"].lower() for c in context_chunks])
        context_words = set(re.findall(r"\w+", combined_context))
        
        overlap = len(answer_words.intersection(context_words)) / len(answer_words)
        is_grounded = overlap >= 0.20
        return is_grounded, round(overlap, 3)
