
import re
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any




@dataclass
class Config:
    """Global configuration for the answer-extraction engine."""
    MIN_RELEVANCE_SCORE: float = 0.08   # below this, treat as "not found"
    MAX_ANSWER_SENTENCES: int = 2       # how many sentences to combine into an answer
    HIGH_CONFIDENCE: float = 0.45
    MEDIUM_CONFIDENCE: float = 0.20
    BM25_K1: float = 1.5
    BM25_B: float = 0.75


CONFIG = Config()



class TextProcessor:
    """Text normalization, tokenization, and sentence splitting."""

    STOPWORDS = frozenset([
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
        'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
        'and', 'but', 'if', 'or', 'this', 'that', 'these', 'those', 'what',
        'which', 'who', 'whom', 'tell', 'give', 'show', 'please', 'me',
        'my', 'i', 'you', 'your',
    ])

    _TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")
    # _SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')
    _SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

    @classmethod
    def normalize(cls, text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        text = text.replace('₹', ' rupees ').replace('%', ' percent ')
        return text.strip()

    @classmethod
    def tokenize(cls, text: str, remove_stopwords: bool = True) -> List[str]:
        tokens = cls._TOKEN_RE.findall(text.lower())
        if remove_stopwords:
            tokens = [t for t in tokens if t not in cls.STOPWORDS]
        return tokens

    @classmethod
    def split_sentences(cls, text: str) -> List[str]:
        flat = re.sub(r'\s+', ' ', text or "").strip()
        if not flat:
            return []
        return [s.strip() for s in cls._SENT_SPLIT_RE.split(flat) if s.strip()]



@dataclass
class Entity:
    value: str
    numeric_value: Optional[float]
    entity_type: str
    confidence: float
    span: Tuple[int, int] = (0, 0)


class EntityExtractor:
    """
    Domain-agnostic entity extraction: currency amounts, percentages,
    dates, durations, and plain numbers. Not tied to any single document
    domain (loans, invoices, contracts, etc. all share these primitives).
    """

    _PATTERNS = [
        ("currency", re.compile(
            r'(?:(?:Rs\.?|INR|₹|\$|USD|€|EUR|£|GBP)\s?[\d,]+(?:\.\d+)?(?:\s?(?:lakh|lakhs|cr|crore|crores|k|million|billion))?'
            r'|[\d,]+(?:\.\d+)?\s?(?:rupees|dollars|lakh|lakhs|crore|crores))',
            re.IGNORECASE)),
        ("percentage", re.compile(r'\d+(?:\.\d+)?\s?%|\d+(?:\.\d+)?\s?percent', re.IGNORECASE)),
        ("date", re.compile(
            r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
            r'|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{2,4}'
            r'|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{2,4})\b',
            re.IGNORECASE)),
        ("duration", re.compile(r'\d+(?:\.\d+)?\s?(?:years?|months?|days?|weeks?|yrs?|mos?)\b', re.IGNORECASE)),
        ("number", re.compile(r'\b\d[\d,]*(?:\.\d+)?\b')),
    ]

    @classmethod
    def extract_all(cls, text: str) -> List[Entity]:
        found: List[Entity] = []
        claimed_spans: List[Tuple[int, int]] = []

        for etype, pattern in cls._PATTERNS:
            for m in pattern.finditer(text):
                span = m.span()
                # Skip if this span overlaps an already-claimed (higher
                # priority, since _PATTERNS is ordered specific->generic) span.
                if any(not (span[1] <= s or span[0] >= e) for s, e in claimed_spans):
                    continue
                claimed_spans.append(span)
                raw = m.group().strip()
                numeric = cls._to_numeric(raw)
                found.append(Entity(
                    value=raw,
                    numeric_value=numeric,
                    entity_type=etype,
                    confidence=0.9 if etype != "number" else 0.5,
                    span=span,
                ))

        found.sort(key=lambda e: e.span[0])
        return found

    @staticmethod
    def _to_numeric(raw: str) -> Optional[float]:
        digits = re.sub(r'[^\d.]', '', raw)
        try:
            return float(digits) if digits else None
        except ValueError:
            return None




class SentenceRanker:
    """Ranks sentences from retrieved chunks by relevance to the query."""

    @staticmethod
    def _bm25_score(query_tokens: List[str], doc_tokens: List[str], avg_len: float) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        doc_len = len(doc_tokens)
        freqs = Counter(doc_tokens)
        score = 0.0
        for term in set(query_tokens):
            f = freqs.get(term, 0)
            if f == 0:
                continue
            idf = math.log(1 + (1.0 / (1 + f)))
            numerator = f * (CONFIG.BM25_K1 + 1)
            denominator = f + CONFIG.BM25_K1 * (
                1 - CONFIG.BM25_B + CONFIG.BM25_B * (doc_len / max(avg_len, 1))
            )
            score += idf * (numerator / denominator)
        return score

    @classmethod
    def rank(cls, query: str, chunks: List[str]) -> List[Tuple[str, float]]:
        """
        Returns (sentence, score) pairs across all sentences in all
        chunks, sorted best-first. Scores are normalized to roughly
        [0, 1] by the max observed score for interpretability.
        """
        query_tokens = TextProcessor.tokenize(query)
        sentences: List[str] = []
        for chunk in chunks:
            sentences.extend(TextProcessor.split_sentences(chunk))
        if not sentences or not query_tokens:
            return []

        tokenized = [TextProcessor.tokenize(s) for s in sentences]
        avg_len = sum(len(t) for t in tokenized) / max(len(tokenized), 1)

        raw_scores = [cls._bm25_score(query_tokens, toks, avg_len) for toks in tokenized]
        max_score = max(raw_scores) if raw_scores else 0.0
        norm_scores = [s / max_score if max_score > 0 else 0.0 for s in raw_scores]

        ranked = list(zip(sentences, norm_scores))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked



class LabelValueExtractor:
    """
    Extracts 'Label : Value' pairs from form/statement-style text
    (e.g. 'Loan Amount Sanctioned : Rs. 200000/-') and matches the
    best label against the user's query. This bypasses fragile
    sentence-boundary detection entirely for structured documents,
    which is far more reliable than picking "top sentences" when the
    source text is really a list of fields, not prose.
    """

    _PATTERN = re.compile(
        r'(?P<label>[A-Z][A-Za-z][A-Za-z\s\-/]{1,45}?)\s*:\s*(?P<value>.+?)'
        r'(?=\s+[A-Z][A-Za-z][A-Za-z\s\-/]{1,45}?\s*:|$)'
    )

    @classmethod
    def best_match(cls, query: str, chunks: List[str]):
        """Returns (label, value, score) for the best-matching field, or None."""
        q_tokens = set(TextProcessor.tokenize(query))
        if not q_tokens:
            return None

        best = None
        best_score = 0.0
        for chunk in chunks:
            for m in cls._PATTERN.finditer(chunk):
                label = m.group('label').strip()
                value = m.group('value').strip()
                l_tokens = set(TextProcessor.tokenize(label))
                if not l_tokens:
                    continue
                union = q_tokens | l_tokens
                score = len(q_tokens & l_tokens) / len(union) if union else 0.0
                if score > best_score:
                    best_score = score
                    best = (label, value)

        if best and best_score >= 0.3:
            return best[0], best[1], best_score
        return None
    

class ProximityExtractor:
    """
    Fallback for table-flattened text (e.g. PDF tables extracted as
    'From Year To Year EMI Amount Dec 2022 Dec 2031 Rs 2546' with no
    colons or sentence punctuation to anchor on). Finds where the
    query's keywords occur in the text, then grabs the nearest value
    of an appropriate type (currency, percentage, date, duration,
    number) that follows.
    """

    _CURRENCY_RE = re.compile(r'(?:Rs\.?|INR|₹|\$)\s?[\d,]+(?:\.\d+)?', re.IGNORECASE)
    _PERCENT_RE = re.compile(r'\d+(?:\.\d+)?\s?%|\d+(?:\.\d+)?\s?percent', re.IGNORECASE)
    _DURATION_RE = re.compile(r'\d+(?:\.\d+)?\s?(?:years?|months?|days?|weeks?|yrs?)\b', re.IGNORECASE)
    _DATE_RE = re.compile(
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{2,4}\b',
        re.IGNORECASE)
    _NUMBER_RE = re.compile(r'\b\d[\d,]*(?:\.\d+)?\b')
    _WINDOW = 50  # characters to look ahead of the matched keyword position

    @classmethod
    def _best_anchor(cls, query: str, text: str) -> int:
        text_lower = text.lower()
        q_lower = query.lower().strip()

        idx = text_lower.find(q_lower)
        if idx != -1:
            return idx + len(q_lower)

        q_tokens = TextProcessor.tokenize(query)
        if not q_tokens:
            return -1

        positions = []
        for tok in set(q_tokens):
            for m in re.finditer(r'\b' + re.escape(tok) + r'\b', text_lower):
                positions.append(m.end())
        if not positions:
            return -1

        def nearby_score(pos):
            window = text_lower[max(0, pos - 40):pos + 40]
            return sum(1 for t in q_tokens if t in window)

        return max(positions, key=nearby_score)

    @classmethod
    def best_match(cls, query: str, chunks: List[str]):
        """Returns (anchor_phrase, value, score) or None."""
        best_value = None
        best_pos_quality = -1

        for chunk in chunks:
            pos = cls._best_anchor(query, chunk)
            if pos == -1:
                continue
            window_text = chunk[pos:pos + cls._WINDOW]
            for pattern in (cls._CURRENCY_RE, cls._PERCENT_RE,
                             cls._DURATION_RE, cls._DATE_RE, cls._NUMBER_RE):
                m = pattern.search(window_text)
                if m:
                    # Prefer matches found closer to the start of the window
                    quality = cls._WINDOW - m.start()
                    if quality > best_pos_quality:
                        best_pos_quality = quality
                        best_value = m.group().strip()
                    break

        if best_value:
            return query.strip(), best_value, 0.6
        return None    

class IntentDetector:
    """
    Lightweight, generic intent labeling for display/logging purposes.
    Unlike the original, this doesn't gate answer extraction — it's
    purely descriptive (e.g. for a UI badge showing "Numeric Lookup" vs
    "General Question"), so it never causes a wrong-domain miss.
    """

    _NUMERIC_CUES = re.compile(
        r'\bhow (much|many|long)\b|\bwhat is the (rate|amount|percentage|price|cost|number)\b',
        re.IGNORECASE)
    _DATE_CUES = re.compile(r'\bwhen\b|\bwhat date\b', re.IGNORECASE)

    @classmethod
    def detect(cls, query: str) -> str:
        if cls._DATE_CUES.search(query or ""):
            return "Date Lookup"
        if cls._NUMERIC_CUES.search(query or ""):
            return "Numeric Lookup"
        if not (query or "").strip():
            return "Unknown"
        return "General Question"




@dataclass
class Answer:
    text: str
    value: str
    intent_label: str
    confidence_band: str
    supporting_sentences: List[str] = field(default_factory=list)
    score_breakdown: Dict[str, Any] = field(default_factory=dict)


class AnswerGenerator:

    @classmethod
    def generate(cls, query: str, chunks: List[str], explain: bool = False) -> Answer:
        intent_label = IntentDetector.detect(query)

        if not query or not query.strip():
            return cls._error_answer("Please provide a question.", intent_label)
        if not chunks:
            return cls._error_answer(
                "No relevant document content was retrieved for this question.",
                intent_label,
            )
        
        # Try structured "Label : Value" matching first — much more
        # reliable than sentence extraction for forms/statements.
        label_match = LabelValueExtractor.best_match(query, chunks)
        # Fall back to proximity matching for table-flattened text
        # (no colons, no clean sentence punctuation nearby).
        prox_match = ProximityExtractor.best_match(query, chunks)
        if prox_match:
            label, value, score = prox_match
            return Answer(
                text=f"{label}: {value}",
                value=value,
                intent_label=intent_label,
                confidence_band=cls._confidence_band(score),
                supporting_sentences=[f"{label}: {value}"],
                score_breakdown={"proximity_match_score": round(score, 3)} if explain else {},
            )

        if label_match:
            label, value, score = label_match
            return Answer(
                text=f"{label} : {value}",
                value=value,
                intent_label=intent_label,
                confidence_band=cls._confidence_band(min(score + 0.3, 1.0)),
                supporting_sentences=[f"{label} : {value}"],
                score_breakdown={"label_match_score": round(score, 3)} if explain else {},
            )

        ranked = SentenceRanker.rank(query, chunks)
        if not ranked or ranked[0][1] < CONFIG.MIN_RELEVANCE_SCORE:
            return cls._error_answer(
                "This information doesn't appear to be present in the uploaded document.",
                intent_label,
            )

        top_sentences = [s for s, score in ranked[:CONFIG.MAX_ANSWER_SENTENCES]
                          if score >= CONFIG.MIN_RELEVANCE_SCORE]
        top_score = ranked[0][1]

        # Prefer surfacing a concrete entity value when the query is a
        # numeric/date lookup and the best sentence contains one.
        value = "N/A"
        if IntentDetector._NUMERIC_CUES.search(query) or IntentDetector._DATE_CUES.search(query):
            entities = EntityExtractor.extract_all(top_sentences[0])
            if entities:
                value = entities[0].value

        confidence_band = cls._confidence_band(top_score)
        answer_text = " ".join(top_sentences)

        breakdown = {"top_score": round(top_score, 3)} if explain else {}

        return Answer(
            text=answer_text,
            value=value,
            intent_label=intent_label,
            confidence_band=confidence_band,
            supporting_sentences=top_sentences,
            score_breakdown=breakdown,
        )

    @staticmethod
    def _confidence_band(score: float) -> str:
        if score >= CONFIG.HIGH_CONFIDENCE:
            return "HIGH"
        if score >= CONFIG.MEDIUM_CONFIDENCE:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _error_answer(msg: str, intent_label: str) -> Answer:
        return Answer(
            text=msg,
            value="N/A",
            intent_label=intent_label,
            confidence_band="LOW",
        )




def generate_answer(query: str, chunks: List[str], explain: bool = False) -> str:
    """
    PRIMARY ENTRY POINT — Generate an answer string from a query and the
    document chunks retrieved for it. Never fabricates: the returned text
    is always either lifted directly from `chunks` or an explicit
    "not found" / error message.
    """
    try:
        answer = AnswerGenerator.generate(query, chunks, explain=explain)
        if explain and answer.score_breakdown:
            breakdown = " | ".join(f"{k}={v}" for k, v in answer.score_breakdown.items())
            return f"{answer.text}  [{answer.confidence_band}] [scores: {breakdown}]"
        return answer.text
    except Exception as e:
        return f"Unable to process query: {e}"


def generate_answer_structured(query: str, chunks: List[str]) -> Answer:
    """Like generate_answer but returns the full Answer dataclass."""
    try:
        return AnswerGenerator.generate(query, chunks, explain=True)
    except Exception as e:
        return AnswerGenerator._error_answer(f"Error: {e}", IntentDetector.detect(query))


def batch_generate(queries: List[str], chunks: List[str]) -> Dict[str, str]:
    """Process multiple queries against the same chunk set."""
    results = {}
    for q in queries:
        try:
            results[q] = generate_answer(q, chunks)
        except Exception as e:
            results[q] = f"Error processing query: {e}"
    return results


def get_intent(query: str) -> str:
    """Get a descriptive intent label for a query. Never returns None."""
    return IntentDetector.detect(query)


def extract_entities(text: str) -> List[Dict[str, Any]]:
    """Extract all generic entities (currency, %, date, duration, number) from text."""
    return [
        {
            'value': e.value,
            'numeric': e.numeric_value,
            'type': e.entity_type,
            'confidence': round(e.confidence, 3),
        }
        for e in EntityExtractor.extract_all(text)
    ]


def chunk_document(text: str, overlap: int = 1) -> List[str]:
    """
    Split a raw document into sentence-level chunks with optional overlap.
    (Kept for API compatibility / standalone use of this module; main.py's
    primary chunking path is utils.smart_chunk.)
    """
    sentences = TextProcessor.split_sentences(text)
    if len(sentences) <= 3:
        return sentences

    chunks = []
    step = max(1, 3 - overlap)
    for i in range(0, len(sentences), step):
        window = sentences[i:i + 3]
        if window:
            chunks.append(' '.join(window))
    return chunks
