
import re
from typing import List

import PyPDF2


_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\u201c])')


_ABBREVIATIONS = (
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "eg", "ie",
    "rs", "inr", "usd", "no", "vol", "fig", "approx", "e.g", "i.e",
)


def extract_text(pdf_path: str) -> str:
    """
    Extract text from a PDF file, page by page.

    Improvements:
    - Skips pages that return None (e.g. scanned/image-only pages) instead
      of crashing with a TypeError.
    - Preserves page boundaries with a double newline so downstream
      chunking doesn't merge unrelated sections/headers together.
    """
    pages_text: List[str] = []
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
            except Exception:
                # A malformed page shouldn't take down the whole extraction.
                page_text = ""
            page_text = page_text.strip()
            if page_text:
                pages_text.append(page_text)

    full_text = "\n\n".join(pages_text)
    return clean_text(full_text)


def clean_text(text: str) -> str:
    """
    Normalize whitespace while preserving paragraph structure.

    Collapses runs of spaces/tabs within a line, but keeps paragraph
    breaks (double newlines) intact so chunking can still reason about
    document structure instead of treating the whole PDF as one blob.
    """
    if not text:
        return ""

    # Collapse horizontal whitespace only.
    text = re.sub(r'[ \t]+', ' ', text)
    # Collapse 3+ newlines down to a paragraph break.
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Fix hyphenated line-break artifacts common in PDF extraction,
    # e.g. "process-\ning" -> "processing".
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    # Trim trailing spaces before newlines.
    text = re.sub(r' +\n', '\n', text)

    return text.strip()


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, guarding against common abbreviations."""
    # Normalize paragraph breaks to spaces for sentence-level splitting;
    # paragraph structure isn't needed once we're chunking by sentence.
    flat = re.sub(r'\s+', ' ', text).strip()
    if not flat:
        return []

    raw_sentences = _SENTENCE_SPLIT_RE.split(flat)

    # Re-merge any split that occurred right after a known abbreviation.
    sentences: List[str] = []
    buffer = ""
    for sent in raw_sentences:
        if buffer:
            buffer = f"{buffer} {sent}"
        else:
            buffer = sent

        last_word = re.sub(r'[.,]', '', buffer.split(" ")[-1]).lower() if buffer else ""
        if last_word in _ABBREVIATIONS:
            continue  # keep accumulating into buffer
        sentences.append(buffer.strip())
        buffer = ""

    if buffer:
        sentences.append(buffer.strip())

    return [s for s in sentences if s]


def smart_chunk(
    text: str,
    max_words: int = 200,
    overlap_sentences: int = 1,
    min_words: int = 20,
) -> List[str]:
    """
    Split text into overlapping, sentence-aware chunks.

    Args:
        text: Cleaned document text.
        max_words: Target maximum words per chunk.
        overlap_sentences: Number of trailing sentences from the previous
            chunk to repeat at the start of the next chunk. This prevents
            answers from being cut in half at a chunk boundary.
        min_words: Minimum words for a trailing chunk; smaller trailing
            chunks are merged into the previous chunk instead of being
            kept as a near-empty fragment.

    Returns:
        List of chunk strings.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_words = 0

    for sent in sentences:
        sent_words = len(sent.split())
        if current and current_words + sent_words > max_words:
            chunks.append(" ".join(current).strip())
            # Carry the last N sentences forward for overlap/context continuity.
            overlap = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current = list(overlap)
            current_words = sum(len(s.split()) for s in current)

        current.append(sent)
        current_words += sent_words

    if current:
        chunks.append(" ".join(current).strip())

    # Merge a too-small trailing chunk into its predecessor so it doesn't
    # get equal retrieval weight against substantive chunks.
    if len(chunks) > 1 and len(chunks[-1].split()) < min_words:
        chunks[-2] = f"{chunks[-2]} {chunks[-1]}".strip()
        chunks.pop()

    return chunks
