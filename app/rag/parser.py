"""
Enterprise Document Parser & Preprocessing Engine
Supports Plain Text, Markdown, Tabular Data (CSV/TSV/JSON), and Scanned Document OCR simulation.
Integrates with RecursiveSemanticChunker for semantic ingestion into PGVector.
"""

import re
import csv
import json
import io
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from app.rag.chunker import RecursiveSemanticChunker


class DocumentType(str, Enum):
    TEXT = "text"
    MARKDOWN = "markdown"
    CSV = "csv"
    TSV = "tsv"
    JSON_TABLE = "json_table"
    OCR_SCANNED = "ocr_scanned"
    AUTO = "auto"


class ParsedDocument:
    """Represents the structured result of an enterprise document parsing operation."""
    def __init__(
        self,
        content: str,
        doc_type: DocumentType,
        metadata: Optional[Dict[str, Any]] = None,
        sections: Optional[List[Dict[str, Any]]] = None,
        raw_text: Optional[str] = None
    ):
        self.content = content
        self.doc_type = doc_type
        self.metadata = metadata or {}
        self.sections = sections or []
        self.raw_text = raw_text or content

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "doc_type": self.doc_type.value if isinstance(self.doc_type, DocumentType) else str(self.doc_type),
            "metadata": self.metadata,
            "sections_count": len(self.sections),
            "char_count": len(self.content)
        }

    def __repr__(self) -> str:
        return f"<ParsedDocument type={self.doc_type.value} chars={len(self.content)} sections={len(self.sections)}>"


class EnterpriseDocumentParser:
    """
    Enterprise Document Ingestion Parser with multi-modal format normalization,
    tabular linearization, and OCR artifact remediation.
    """

    # Common OCR substitution errors and artifacts
    OCR_REPLACEMENTS = [
        (r"(?<=\w)-\n(?=\w)", ""),             # Line-break hyphenation stitch (e.g. "securi-\nty" -> "security")
        (r"\x0c", "\n--- Page Break ---\n"),  # Form feed / page separator
        (r"[^\x00-\x7F]+", " "),               # Replace non-ASCII noise chars if any corrupted
        (r"\bO([0-9]{3,})\b", r"0\1"),         # Leading 'O' mistyped for zero in numeric codes
        (r"\b([0-9]+)O\b", r"\g<1>0"),         # Trailing 'O' mistyped for zero
        (r"(\r\n|\r)", "\n"),                  # Carriage return normalization
        (r"[ \t]+", " "),                      # Whitespace compaction
    ]

    def __init__(self, default_chunker: Optional[RecursiveSemanticChunker] = None):
        self.default_chunker = default_chunker or RecursiveSemanticChunker()

    def detect_doc_type(self, content: str, filename: Optional[str] = None) -> DocumentType:
        """Heuristically detects document type based on extension and content inspection."""
        if filename:
            ext = Path(filename).suffix.lower()
            if ext in [".md", ".markdown"]:
                return DocumentType.MARKDOWN
            if ext == ".csv":
                return DocumentType.CSV
            if ext == ".tsv":
                return DocumentType.TSV
            if ext in [".json", ".jsonl"]:
                return DocumentType.JSON_TABLE
            if ext in [".ocr", ".scan", ".scanned"]:
                return DocumentType.OCR_SCANNED
            if ext in [".txt", ".log", ".text"]:
                return DocumentType.TEXT

        stripped = content.strip()
        if not stripped:
            return DocumentType.TEXT

        # Check JSON Table / Array
        if (stripped.startswith("[") and stripped.endswith("]")) or (stripped.startswith("{") and stripped.endswith("}")):
            try:
                parsed_json = json.loads(stripped)
                if isinstance(parsed_json, list) or isinstance(parsed_json, dict):
                    return DocumentType.JSON_TABLE
            except Exception:
                pass

        # Check OCR signature (e.g., explicit OCR header or bounding-box tags or page breaks)
        if "[OCR_CONFIDENCE:" in stripped or "[PAGE_SCAN_" in stripped or "\x0c" in content or "--- Page Break ---" in content:
            return DocumentType.OCR_SCANNED

        # Check Markdown features (headers, blockquotes, markdown tables)
        if re.search(r"^#{1,6}\s+\w+", stripped, re.MULTILINE) or re.search(r"\|.+\|.+\|", stripped):
            return DocumentType.MARKDOWN

        # Check CSV features (multiple lines with consistent comma counts)
        lines = [line.strip() for line in stripped.split("\n") if line.strip()]
        if len(lines) >= 2:
            comma_counts = [line.count(",") for line in lines[:5]]
            if comma_counts[0] > 1 and len(set(comma_counts)) == 1:
                return DocumentType.CSV
            tsv_counts = [line.count("\t") for line in lines[:5]]
            if tsv_counts[0] > 1 and len(set(tsv_counts)) == 1:
                return DocumentType.TSV

        return DocumentType.TEXT

    def parse(
        self,
        content: str,
        doc_type: Union[DocumentType, str] = DocumentType.AUTO,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ParsedDocument:
        """
        Parses raw text content into a standardized ParsedDocument structure.
        """
        meta = dict(metadata or {})
        if filename:
            meta["filename"] = filename

        if isinstance(doc_type, str):
            try:
                resolved_type = DocumentType(doc_type.lower())
            except ValueError:
                resolved_type = DocumentType.AUTO
        else:
            resolved_type = doc_type

        if resolved_type == DocumentType.AUTO:
            resolved_type = self.detect_doc_type(content, filename)

        if resolved_type == DocumentType.MARKDOWN:
            return self._parse_markdown(content, meta)
        elif resolved_type in [DocumentType.CSV, DocumentType.TSV]:
            delimiter = "\t" if resolved_type == DocumentType.TSV else ","
            return self._parse_tabular_csv(content, delimiter=delimiter, metadata=meta, doc_type=resolved_type)
        elif resolved_type == DocumentType.JSON_TABLE:
            return self._parse_tabular_json(content, meta)
        elif resolved_type == DocumentType.OCR_SCANNED:
            return self._parse_ocr_scanned(content, meta)
        else:
            return self._parse_text(content, meta)

    def parse_file(
        self,
        file_path: Union[str, Path],
        doc_type: Union[DocumentType, str] = DocumentType.AUTO,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ParsedDocument:
        """Reads and parses a document from the local filesystem."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        meta = dict(metadata or {})
        meta["file_path"] = str(path.resolve())
        return self.parse(content, doc_type=doc_type, filename=path.name, metadata=meta)

    def parse_and_chunk(
        self,
        content: str,
        chunker: Optional[RecursiveSemanticChunker] = None,
        doc_type: Union[DocumentType, str] = DocumentType.AUTO,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **chunker_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Parses document and executes recursive semantic chunking with unified metadata tagging.
        """
        active_chunker = chunker or (RecursiveSemanticChunker(**chunker_kwargs) if chunker_kwargs else self.default_chunker)
        parsed_doc = self.parse(content, doc_type=doc_type, filename=filename, metadata=metadata)
        
        chunk_meta = {
            **parsed_doc.metadata,
            "doc_type": parsed_doc.doc_type.value if isinstance(parsed_doc.doc_type, DocumentType) else str(parsed_doc.doc_type),
            "total_sections": len(parsed_doc.sections)
        }
        
        return active_chunker.chunk_text(parsed_doc.content, metadata=chunk_meta)

    # ------------------ Internal Format Parsers ------------------

    def _parse_text(self, content: str, metadata: Dict[str, Any]) -> ParsedDocument:
        normalized = "\n".join([line.rstrip() for line in content.splitlines()])
        normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
        paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
        
        sections = [{"title": f"Paragraph {i+1}", "content": p} for i, p in enumerate(paragraphs)]
        metadata["format"] = "plain_text"
        metadata["paragraph_count"] = len(paragraphs)

        return ParsedDocument(
            content=normalized,
            doc_type=DocumentType.TEXT,
            metadata=metadata,
            sections=sections,
            raw_text=content
        )

    def _parse_markdown(self, content: str, metadata: Dict[str, Any]) -> ParsedDocument:
        lines = content.splitlines()
        sections: List[Dict[str, Any]] = []
        current_section = {"title": "Introduction", "content": []}
        
        for line in lines:
            header_match = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
            if header_match:
                if current_section["content"]:
                    sections.append({
                        "title": current_section["title"],
                        "content": "\n".join(current_section["content"]).strip()
                    })
                current_section = {
                    "title": header_match.group(2).strip(),
                    "level": len(header_match.group(1)),
                    "content": []
                }
            else:
                current_section["content"].append(line)

        if current_section["content"]:
            sections.append({
                "title": current_section["title"],
                "content": "\n".join(current_section["content"]).strip()
            })

        normalized_content = "\n\n".join(
            [f"## {s['title']}\n{s['content']}" if s["title"] != "Introduction" else s["content"]
             for s in sections if s["content"]]
        ).strip()

        metadata["format"] = "markdown"
        metadata["headings_count"] = len(sections)

        return ParsedDocument(
            content=normalized_content or content.strip(),
            doc_type=DocumentType.MARKDOWN,
            metadata=metadata,
            sections=sections,
            raw_text=content
        )

    def _parse_tabular_csv(
        self,
        content: str,
        delimiter: str = ",",
        metadata: Optional[Dict[str, Any]] = None,
        doc_type: DocumentType = DocumentType.CSV
    ) -> ParsedDocument:
        meta = dict(metadata or {})
        try:
            reader = csv.reader(io.StringIO(content.strip()), delimiter=delimiter)
            rows = [r for r in reader if r]
        except Exception as e:
            # Fallback to plain text on corrupt csv
            meta["csv_parse_error"] = str(e)
            return self._parse_text(content, meta)

        if not rows:
            return ParsedDocument(content="", doc_type=doc_type, metadata=meta, sections=[])

        headers = [h.strip() for h in rows[0]]
        data_rows = rows[1:]
        
        linearized_lines: List[str] = []
        sections: List[Dict[str, Any]] = []

        for idx, row in enumerate(data_rows):
            row_items = []
            for h_idx, val in enumerate(row):
                header_name = headers[h_idx] if h_idx < len(headers) else f"Column_{h_idx+1}"
                row_items.append(f"{header_name}: {val.strip()}")
            
            row_repr = f"Record {idx + 1} -> " + "; ".join(row_items)
            linearized_lines.append(row_repr)
            sections.append({"title": f"Row {idx + 1}", "content": row_repr})

        normalized_content = (
            f"Dataset Table with Columns [{', '.join(headers)}]:\n\n" +
            "\n".join(linearized_lines)
        )

        meta["format"] = "tabular"
        meta["columns"] = headers
        meta["row_count"] = len(data_rows)

        return ParsedDocument(
            content=normalized_content,
            doc_type=doc_type,
            metadata=meta,
            sections=sections,
            raw_text=content
        )

    def _parse_tabular_json(self, content: str, metadata: Dict[str, Any]) -> ParsedDocument:
        try:
            data = json.loads(content.strip())
        except Exception as e:
            metadata["json_parse_error"] = str(e)
            return self._parse_text(content, metadata)

        sections: List[Dict[str, Any]] = []
        linearized_records: List[str] = []

        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    fields = [f"{k}: {v}" for k, v in item.items()]
                    line = f"Record {i + 1}: " + ", ".join(fields)
                else:
                    line = f"Item {i + 1}: {item}"
                linearized_records.append(line)
                sections.append({"title": f"Item {i+1}", "content": line})
            metadata["record_count"] = len(data)
        elif isinstance(data, dict):
            for k, v in data.items():
                line = f"{k}: {json.dumps(v) if isinstance(v, (dict, list)) else v}"
                linearized_records.append(line)
                sections.append({"title": k, "content": line})
            metadata["key_count"] = len(data)

        normalized_content = "\n".join(linearized_records)
        metadata["format"] = "json_table"

        return ParsedDocument(
            content=normalized_content,
            doc_type=DocumentType.JSON_TABLE,
            metadata=metadata,
            sections=sections,
            raw_text=content
        )

    def _parse_ocr_scanned(self, content: str, metadata: Dict[str, Any]) -> ParsedDocument:
        """
        Cleans and reconstructs text extracted from scanned documents/OCR engines.
        Removes scanning artifacts, stitches broken line hyphenations, extracts confidence scores.
        """
        raw = content
        
        # Extract embedded OCR confidence if present (e.g., "[OCR_CONFIDENCE: 94.8%]")
        conf_match = re.search(r"\[OCR_CONFIDENCE:\s*([\d\.]+)%?\]", raw, re.IGNORECASE)
        ocr_confidence = float(conf_match.group(1)) if conf_match else 92.5
        metadata["ocr_confidence"] = ocr_confidence
        metadata["format"] = "ocr_scanned"

        # Remove explicit confidence tags from clean output
        cleaned = re.sub(r"\[OCR_CONFIDENCE:\s*[\d\.]+%?\]", "", raw)
        
        # Extract page boundaries
        pages = re.split(r"(?:\x0c|\n?---\s*Page Break\s*---\n?|\[PAGE_SCAN_\d+\])", cleaned)
        pages = [p.strip() for p in pages if p.strip()]

        sections: List[Dict[str, Any]] = []
        cleaned_pages: List[str] = []

        for page_idx, page_text in enumerate(pages or [cleaned]):
            # Apply regex replacements for OCR cleanup
            proc_text = page_text
            for pattern, repl in self.OCR_REPLACEMENTS:
                proc_text = re.sub(pattern, repl, proc_text)

            # Clean repetitive line scan artifacts (e.g. repeated underscores or dots)
            proc_text = re.sub(r"[_\.]{4,}", " ", proc_text)
            
            # Reconstruct paragraphs
            paragraphs = [p.strip() for p in proc_text.split("\n\n") if p.strip()]
            page_clean = "\n\n".join(paragraphs)
            
            if page_clean:
                cleaned_pages.append(f"[Page {page_idx + 1}]\n{page_clean}")
                sections.append({
                    "title": f"Page {page_idx + 1}",
                    "page_number": page_idx + 1,
                    "content": page_clean
                })

        normalized_content = "\n\n".join(cleaned_pages).strip()
        metadata["page_count"] = len(cleaned_pages)
        metadata["is_ocr_remediated"] = True

        return ParsedDocument(
            content=normalized_content or cleaned.strip(),
            doc_type=DocumentType.OCR_SCANNED,
            metadata=metadata,
            sections=sections,
            raw_text=raw
        )
