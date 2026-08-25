from typing import List, Dict, Any

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
