import re
from typing import List, Dict, Any, Optional

class RecursiveSemanticChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        metadata = metadata or {}
        if not text:
            return []

        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_len = len(para)
            if current_len + para_len > self.chunk_size and current_chunk:
                chunk_str = "\n\n".join(current_chunk)
                chunks.append({
                    "content": chunk_str,
                    "metadata": {**metadata, "chunk_index": len(chunks)}
                })
                current_chunk = [para]
                current_len = para_len
            else:
                current_chunk.append(para)
                current_len += para_len

        if current_chunk:
            chunk_str = "\n\n".join(current_chunk)
            chunks.append({
                "content": chunk_str,
                "metadata": {**metadata, "chunk_index": len(chunks)}
            })

        return chunks


class StructureAwareChunker(RecursiveSemanticChunker):
    """
    Markdown/heading-aware chunker.

    Splits text on markdown headings (``#``/``##``/``###``) and blank lines,
    hard-splits any single paragraph that exceeds ``chunk_size``, and tags each
    chunk with its originating ``heading`` and a global ``chunk_index``.

    Supports parent/child relationships via :meth:`chunk_with_parents`: consecutive
    chunks that share a heading are grouped under a shared ``parent_id``
    (``"{doc_id}_{heading_slug}"``). Standard :meth:`chunk_text`` returns chunks
    with ``metadata["parent_id"] == None``.
    """

    _HEADING_RE = re.compile(r"^(\s*)(#{1,3})\s+(.+?)\s*$")

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _heading_level(line: str) -> Optional[int]:
        m = StructureAwareChunker._HEADING_RE.match(line)
        return len(m.group(2)) if m else None

    @staticmethod
    def _heading_text(line: str) -> Optional[str]:
        m = StructureAwareChunker._HEADING_RE.match(line)
        return m.group(3).strip() if m else None

    @staticmethod
    def _slugify(text: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9]+", "-", text.strip().lower()).strip("-")
        return slug or "section"

    def _hard_split(self, text: str) -> List[str]:
        """Splits ``text`` into pieces each <= chunk_size chars (best-effort on words)."""
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]
        pieces: List[str] = []
        remaining = text
        while len(remaining) > self.chunk_size:
            window = remaining[: self.chunk_size]
            # Try to break on the last space within the window (>= half-way).
            space = window.rfind(" ")
            if space >= self.chunk_size // 2:
                cut = space + 1
            else:
                cut = self.chunk_size
            piece = remaining[:cut].strip()
            pieces.append(piece)
            remaining = remaining[cut:].lstrip()
        if remaining.strip():
            pieces.append(remaining.strip())
        return pieces

    # ------------------------------------------------------------ structure
    def _scan_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Splits raw text into (heading, paragraphs) sections.

        A heading line starts a new section; subsequent non-heading lines are the
        body. Within the body, blank lines separate paragraphs. Content before the
        first heading is treated as a section with ``heading is None``.
        """
        body_buffer: List[str] = []
        paragraphs_buffer: List[str] = []
        sections: List[Dict[str, Any]] = []

        def flush_paragraphs() -> None:
            if body_buffer:
                paragraphs_buffer.append("\n".join(body_buffer).strip())
                body_buffer.clear()

        def close_section(heading: Optional[str]) -> None:
            flush_paragraphs()
            if paragraphs_buffer or heading is not None:
                sections.append({
                    "heading": heading,
                    "paragraphs": [p for p in paragraphs_buffer if p],
                })
            paragraphs_buffer.clear()

        lines = text.split("\n")
        current_heading: Optional[str] = None
        seen_heading = False

        for raw in lines:
            line = raw.rstrip()
            level = self._heading_level(line)
            if level is not None:
                close_section(current_heading)
                current_heading = self._heading_text(line)
                seen_heading = True
                continue
            if line.strip() == "":
                flush_paragraphs()
                continue
            body_buffer.append(line.strip())

        # Trailing content.
        if current_heading is not None or seen_heading:
            close_section(current_heading)
        elif body_buffer or paragraphs_buffer:
            bodies = list(body_buffer) + list(paragraphs_buffer)
            sections.append({"heading": None, "paragraphs": [b for b in bodies if b]})

        return sections

    def _chunk_text_with_meta(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Core implementation shared by chunk_text / chunk_with_parents."""
        metadata = metadata or {}
        if not text:
            return []

        sections = self._scan_sections(text)
        chunks: List[Dict[str, Any]] = []
        chunk_index = 0

        for section in sections:
            heading = section["heading"]
            paragraphs = section["paragraphs"]

            # A heading with no body still yields a (heading-only) section marker
            # when explicitly present (e.g. standalone "# Title").
            if heading is not None and not paragraphs:
                paragraphs = [""]

            for para in paragraphs:
                for piece in self._hard_split(para) if para else [""]:
                    meta = {**metadata, "chunk_index": chunk_index}
                    if heading is not None:
                        meta["heading"] = heading
                    meta["parent_id"] = None  # structured parent linkage added later
                    chunks.append({"content": piece, "metadata": meta})
                    chunk_index += 1

        return chunks

    def _attach_parent_ids(
        self, chunks: List[Dict[str, Any]], metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Re-keys consecutive chunks sharing a heading under one parent_id."""
        doc_id = str(
            metadata.get("doc_id")
            or metadata.get("document_id")
            or metadata.get("source")
            or "doc"
        )
        parent_for_heading: Dict[Optional[str], str] = {}

        out = []
        for chunk in chunks:
            heading = chunk["metadata"].get("heading")
            meta = dict(chunk["metadata"])
            if heading is not None and heading not in parent_for_heading:
                parent_for_heading[heading] = f"{doc_id}_{self._slugify(heading)}"
            meta["parent_id"] = parent_for_heading.get(heading)
            out.append({"content": chunk["content"], "metadata": meta})
        return out

    # -------------------------------------------------------------- public API
    def chunk_text(
        self, text: str, metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Structure-aware chunking. Same signature/output format as the parent
        :meth:`RecursiveSemanticChunker.chunk_text` (list of ``{content, metadata}``
        dicts), but splits on markdown headings/blank lines and records
        ``metadata["heading"]`` and ``metadata["chunk_index"]``.
        """
        return self._chunk_text_with_meta(text, metadata or {})

    def chunk_with_parents(
        self, text: str, metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Like :meth:`chunk_text` but additionally links consecutive chunks under the
        same heading to a shared ``metadata["parent_id"]`` of the form
        ``"{doc_id}_{heading_slug}"`` (default ``None`` when no heading).
        """
        metadata = metadata or {}
        chunks = self._chunk_text_with_meta(text, metadata)
        return self._attach_parent_ids(chunks, metadata)
