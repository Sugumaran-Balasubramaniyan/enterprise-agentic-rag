import re
from typing import Dict, Any, List, Optional


class CitationVerifierTool:
    """
    Analyzes generated answer against source context chunks, verifies that claims
    have corresponding citations/excerpts, and computes citation precision and coverage scores.
    """
    name = "citation_verifier"
    description = (
        "Verifies factual claims in generated answers against retrieved source chunks "
        "and calculates citation precision and coverage scores."
    )

    def __init__(self, overlap_threshold: float = 0.20):
        self.overlap_threshold = overlap_threshold

    @staticmethod
    def _extract_claims(text: str) -> List[str]:
        """Splits answer into discrete factual claim sentences, stripping formatting."""
        if not text:
            return []

        # Split on sentence boundaries and newlines
        raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        claims = []

        for s in raw_sentences:
            s_clean = s.strip()
            # Skip empty, headers, markdown bullets only, or very short sentences
            if len(s_clean) < 15:
                continue
            if s_clean.startswith("#") or s_clean.startswith("```"):
                continue

            # Strip bullet prefixes
            s_clean = re.sub(r"^[\*\-\•\d+\.]\s*", "", s_clean).strip()
            if len(s_clean) < 15:
                continue

            claims.append(s_clean)

        return claims

    @staticmethod
    def _extract_citation_tags(claim: str) -> List[str]:
        """Extracts explicit bracketed citations from a claim, e.g. [doc_arch_001], [1], [doc.pdf]."""
        matches = re.findall(r"\[([^\]]+)\]", claim)
        return [m.strip() for m in matches if m.strip()]

    def verify(self, answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synchronously verifies claims against source chunks.
        """
        if not answer or not answer.strip():
            return {
                "verified": True,
                "citation_precision": 1.0,
                "citation_coverage": 1.0,
                "precision": 1.0,
                "coverage": 1.0,
                "total_claims": 0,
                "verified_claims_count": 0,
                "unverified_claims_count": 0,
                "verified_claims": [],
                "unverified_claims": [],
                "citations_found": [],
                "summary": "No claims to verify in empty answer."
            }

        if not sources:
            claims = self._extract_claims(answer)
            return {
                "verified": len(claims) == 0,
                "citation_precision": 0.0 if claims else 1.0,
                "citation_coverage": 0.0 if claims else 1.0,
                "precision": 0.0 if claims else 1.0,
                "coverage": 0.0 if claims else 1.0,
                "total_claims": len(claims),
                "verified_claims_count": 0,
                "unverified_claims_count": len(claims),
                "verified_claims": [],
                "unverified_claims": claims,
                "citations_found": [],
                "summary": "No source context provided for verification."
            }

        claims = self._extract_claims(answer)
        if not claims:
            return {
                "verified": True,
                "citation_precision": 1.0,
                "citation_coverage": 1.0,
                "precision": 1.0,
                "coverage": 1.0,
                "total_claims": 0,
                "verified_claims_count": 0,
                "unverified_claims_count": 0,
                "verified_claims": [],
                "unverified_claims": [],
                "citations_found": [],
                "summary": "No distinct claim sentences identified."
            }

        # Tokenize source chunks
        processed_sources = []
        for i, src in enumerate(sources):
            content = str(src.get("content", ""))
            meta = src.get("metadata", {}) or {}
            src_id = str(src.get("id") or f"src_{i}")
            doc_id = str(src.get("document_id") or meta.get("source", ""))
            title = str(meta.get("title", ""))

            tokens = set(re.findall(r"\w+", content.lower()))
            processed_sources.append({
                "index": i + 1,
                "id": src_id,
                "document_id": doc_id,
                "title": title,
                "source_file": str(meta.get("source", "")),
                "content": content,
                "tokens": tokens
            })

        verified_claims = []
        unverified_claims = []
        all_citations = []
        total_precision_score = 0.0

        for claim in claims:
            claim_clean = re.sub(r"\[[^\]]+\]", "", claim)
            claim_tokens = [w.lower() for w in re.findall(r"\w+", claim_clean) if len(w) > 2]
            claim_token_set = set(claim_tokens)

            explicit_tags = self._extract_citation_tags(claim)
            all_citations.extend(explicit_tags)

            best_match_src = None
            best_overlap = 0.0
            citation_valid = False

            if not claim_token_set:
                # Trivial claim without substantial words
                verified_claims.append({
                    "claim": claim,
                    "is_verified": True,
                    "matched_source_id": None,
                    "overlap_score": 1.0,
                    "explicit_citations": explicit_tags,
                    "citation_valid": True
                })
                total_precision_score += 1.0
                continue

            # Check matching against each source chunk
            for src in processed_sources:
                overlap_count = len(claim_token_set.intersection(src["tokens"]))
                overlap_ratio = overlap_count / len(claim_token_set)

                # Check if tag explicitly references this source
                tag_matches = False
                for tag in explicit_tags:
                    tag_lower = tag.lower()
                    if (tag_lower in src["id"].lower() or 
                        tag_lower in src["document_id"].lower() or 
                        tag_lower in src["source_file"].lower() or
                        tag_lower in src["title"].lower() or
                        tag == str(src["index"])):
                        tag_matches = True
                        break

                if tag_matches and overlap_ratio >= (self.overlap_threshold * 0.7):
                    citation_valid = True

                if overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_match_src = src

            is_verified = (best_overlap >= self.overlap_threshold) or citation_valid

            claim_record = {
                "claim": claim,
                "is_verified": is_verified,
                "matched_source_id": best_match_src["id"] if best_match_src else None,
                "matched_document": best_match_src["document_id"] if best_match_src else None,
                "overlap_score": round(best_overlap, 3),
                "explicit_citations": explicit_tags,
                "citation_valid": citation_valid or (is_verified and len(explicit_tags) == 0)
            }

            if is_verified:
                verified_claims.append(claim_record)
                total_precision_score += best_overlap
            else:
                unverified_claims.append(claim)

        total_claims = len(claims)
        coverage = len(verified_claims) / total_claims if total_claims > 0 else 1.0
        precision = (total_precision_score / len(verified_claims)) if len(verified_claims) > 0 else 0.0

        # Bound precision between 0.0 and 1.0
        precision = min(max(precision, 0.0), 1.0)
        overall_verified = coverage >= 0.50

        return {
            "verified": overall_verified,
            "citation_precision": round(precision, 3),
            "citation_coverage": round(coverage, 3),
            "precision": round(precision, 3),
            "coverage": round(coverage, 3),
            "total_claims": total_claims,
            "verified_claims_count": len(verified_claims),
            "unverified_claims_count": len(unverified_claims),
            "verified_claims": verified_claims,
            "unverified_claims": unverified_claims,
            "citations_found": list(set(all_citations)),
            "summary": (
                f"Verified {len(verified_claims)}/{total_claims} claims "
                f"(Coverage: {coverage:.1%}, Precision: {precision:.1%})"
            )
        }

    async def execute(self, answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Asynchronous execution entry point for agent tool dispatch."""
        return self.verify(answer, sources)
