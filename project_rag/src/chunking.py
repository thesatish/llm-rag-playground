import re
import hashlib
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

@dataclass
class Chunk:
    """Represents an enriched text chunk with metadata."""
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata
        }


def estimate_token_count(text: str) -> int:
    """Approximates token count using typical English word-to-token ratio (~0.75 words per token)."""
    words = len(text.split())
    # Typically 1 word is roughly 1.3 to 1.33 tokens
    return max(1, int(words * 1.33))


class FixedSizeChunker:
    """
    Topic 1: Understanding Chunking and Overlap in Advanced RAG.
    Implements a sliding window chunking mechanism with exact character overlap.
    Stride = chunk_size - chunk_overlap.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"Overlap ({chunk_overlap}) must be strictly less than chunk size ({chunk_size})")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.stride = chunk_size - chunk_overlap

    def chunk(self, text: str, doc_id: str, source_name: str = "unknown") -> List[Chunk]:
        chunks: List[Chunk] = []
        text_len = len(text)
        if text_len == 0:
            return chunks

        start = 0
        chunk_idx = 0

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_id = f"{doc_id}_chunk_{chunk_idx:04d}"
                meta = {
                    "doc_id": doc_id,
                    "source": source_name,
                    "chunk_index": chunk_idx,
                    "char_start": start,
                    "char_end": end,
                    "char_length": len(chunk_text),
                    "token_count": estimate_token_count(chunk_text),
                    "chunk_strategy": "fixed_sliding_window",
                    "overlap": self.chunk_overlap,
                    "created_at": time.time()
                }
                chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc_id, text=chunk_text, metadata=meta))
                chunk_idx += 1

            if end >= text_len:
                break
            start += self.stride

        # Populate total_chunks in metadata for backward-forward navigation
        for c in chunks:
            c.metadata["total_chunks"] = len(chunks)

        return chunks


class RecursiveCharacterChunker:
    """
    Topic 1 & 2: Recursive Character Chunking with Metadata.
    Splits text hierarchically using a priority list of separators:
    1. Paragraphs ("\n\n")
    2. Sentences ("\n", ". ")
    3. Clauses ("; ", ", ")
    4. Words (" ")
    5. Characters ("")
    Preserves semantic coherence before resorting to finer splits.
    """
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits: List[str] = []
        for s in splits:
            if not s:
                continue
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if new_separators:
                    sub_splits = self._split_text(s, new_separators)
                    good_splits.extend(sub_splits)
                else:
                    # Hard truncate if no finer separators remain
                    for idx in range(0, len(s), self.chunk_size - self.chunk_overlap):
                        good_splits.append(s[idx:idx + self.chunk_size])

        # Merge splits with overlap
        merged: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for piece in good_splits:
            piece_len = len(piece) + (len(separator) if current_chunk else 0)
            if current_len + piece_len <= self.chunk_size:
                current_chunk.append(piece)
                current_len += piece_len
            else:
                if current_chunk:
                    merged.append(separator.join(current_chunk).strip())
                # Handle overlap by keeping tail pieces
                while current_chunk and current_len > self.chunk_overlap:
                    removed = current_chunk.pop(0)
                    current_len -= (len(removed) + len(separator))
                current_chunk.append(piece)
                current_len += piece_len

        if current_chunk:
            merged.append(separator.join(current_chunk).strip())

        return [m for m in merged if m]

    def chunk(self, text: str, doc_id: str, source_name: str = "unknown", extra_meta: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        raw_chunks = self._split_text(text, self.separators)
        chunks: List[Chunk] = []

        total = len(raw_chunks)
        running_char = 0

        for i, chunk_text in enumerate(raw_chunks):
            # Compute character offsets if possible
            start_pos = text.find(chunk_text[:30], running_char) if len(chunk_text) >= 30 else running_char
            if start_pos == -1:
                start_pos = running_char
            end_pos = start_pos + len(chunk_text)
            running_char = max(0, end_pos - self.chunk_overlap)

            chunk_id = f"{doc_id}_c{i:03d}"
            meta = {
                "doc_id": doc_id,
                "source": source_name,
                "chunk_index": i,
                "total_chunks": total,
                "char_start": start_pos,
                "char_end": end_pos,
                "token_count": estimate_token_count(chunk_text),
                "strategy": "recursive_character",
                "overlap": self.chunk_overlap,
                "created_at": time.time(),
                **(extra_meta or {})
            }
            chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc_id, text=chunk_text, metadata=meta))

        return chunks


class MarkdownSemanticChunker:
    """
    Topic 2: Building a Document Chunking Strategy with Metadata.
    Splits Markdown by structural headers (#, ##, ###) and preserves
    hierarchical breadcrumb metadata (e.g. 'Architecture > Vector Store > HNSW').
    """
    def __init__(self, max_chunk_size: int = 600, overlap: int = 80):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.recursive_chunker = RecursiveCharacterChunker(
            chunk_size=max_chunk_size,
            chunk_overlap=overlap
        )

    def chunk(self, markdown_text: str, doc_id: str, source_name: str = "unknown") -> List[Chunk]:
        # Split into lines and detect markdown headers
        lines = markdown_text.splitlines()
        sections: List[Dict[str, Any]] = []
        current_headers: List[str] = ["Introduction"]
        current_buffer: List[str] = []

        header_regex = re.compile(r"^(#{1,6})\s+(.+)$")

        for line in lines:
            match = header_regex.match(line)
            if match:
                # Flush previous section
                if current_buffer:
                    sections.append({
                        "section_path": " > ".join(current_headers),
                        "heading": current_headers[-1],
                        "text": "\n".join(current_buffer).strip()
                    })
                    current_buffer = []

                level = len(match.group(1))
                title = match.group(2).strip()

                # Adjust header hierarchy depth
                while len(current_headers) >= level:
                    current_headers.pop()
                current_headers.append(title)
            else:
                current_buffer.append(line)

        if current_buffer:
            sections.append({
                "section_path": " > ".join(current_headers),
                "heading": current_headers[-1],
                "text": "\n".join(current_buffer).strip()
            })

        # Now chunk each section if it exceeds max_chunk_size
        all_chunks: List[Chunk] = []
        chunk_idx = 0

        for sec in sections:
            sec_text = sec["text"]
            if not sec_text:
                continue

            sec_meta = {
                "section_path": sec["section_path"],
                "section_heading": sec["heading"]
            }

            if len(sec_text) <= self.max_chunk_size:
                cid = f"{doc_id}_c{chunk_idx:03d}"
                meta = {
                    "doc_id": doc_id,
                    "source": source_name,
                    "chunk_index": chunk_idx,
                    "token_count": estimate_token_count(sec_text),
                    "strategy": "markdown_section",
                    "overlap": 0,
                    "created_at": time.time(),
                    **sec_meta
                }
                all_chunks.append(Chunk(chunk_id=cid, doc_id=doc_id, text=sec_text, metadata=meta))
                chunk_idx += 1
            else:
                sub_chunks = self.recursive_chunker.chunk(
                    sec_text,
                    doc_id=doc_id,
                    source_name=source_name,
                    extra_meta=sec_meta
                )
                for sc in sub_chunks:
                    sc.chunk_id = f"{doc_id}_c{chunk_idx:03d}"
                    sc.metadata["chunk_index"] = chunk_idx
                    all_chunks.append(sc)
                    chunk_idx += 1

        for c in all_chunks:
            c.metadata["total_chunks"] = len(all_chunks)

        return all_chunks

