"""Shared, dependency-free tokenization for the lexical side of Hybrid
Retrieval — `BM25Retriever` (candidate search) and `LexicalReranker`
(relevance scoring) both need the same word-level notion of a "term", so
it exists in exactly one place rather than drifting between the two.

Deliberately not `tiktoken` (subword/BPE tokens, built for counting
OpenAI billing units, not for matching whole words like "Contact" or
"C_10") and not a full stemmer/NLP dependency — a small fixed stopword
list plus light suffix-stripping is enough to recover common
inflections ("contacts" -> "contact") without adding a new dependency.
"""
import re

_WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+")

# Common English function words that carry no topical relevance signal
# on their own — filtered out before overlap/IDF scoring so a shared
# "how does the" doesn't count as a term match. Deliberately small and
# conservative: better to under-filter (an extra low-weight term) than
# to accidentally strip a real content word.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "how", "what", "when", "where", "why", "which", "who", "whom",
        "do", "does", "did", "doing",
        "of", "in", "on", "at", "to", "for", "with", "by", "from", "as",
        "and", "or", "but", "if", "so", "than", "that", "this", "these", "those",
        "it", "its", "i", "you", "your", "we", "our", "they", "their", "he", "she",
        "can", "could", "should", "would", "will", "shall", "may", "might", "must",
        "not", "no", "yes", "about", "into", "over", "under", "again",
    }
)

_SUFFIXES = ("ing", "ed", "es", "s")


def _stem(term: str) -> str:
    """Strips one common suffix — enough to match "contacts" with
    "contact" or "merging" with "merge"-ish forms, without a real
    stemmer. Never strips below 3 characters, so short, meaningful
    tokens (e.g. an id like "c10") are never mangled.
    """
    for suffix in _SUFFIXES:
        if term.endswith(suffix) and len(term) - len(suffix) >= 3:
            return term[: -len(suffix)]
    return term


def tokenize(text: str) -> list[str]:
    """Lowercased, stopword-filtered, lightly stemmed word tokens — used
    identically for BM25's corpus/query text and for the reranker's
    term-overlap scoring, so "does this term match" means the same thing
    in both stages.
    """
    tokens = [match.group(0).lower() for match in _WORD_PATTERN.finditer(text)]
    tokens = [token for token in tokens if token not in _STOPWORDS]
    return [_stem(token) for token in tokens]
