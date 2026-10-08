"""WER and domain-term recall, with the same text normalisation on both sides."""
import re
import jiwer

_norm = None


def normalize(text: str) -> str:
    global _norm
    try:
        if _norm is None:
            from whisper_normalizer.english import EnglishTextNormalizer
            _norm = EnglishTextNormalizer()
        return _norm(text)
    except ImportError:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", text.lower())).strip()


def wer(ref: str, hyp: str) -> float:
    r, h = normalize(ref), normalize(hyp)
    return jiwer.wer(r, h) if r else float("nan")


def term_recall(terms, hyp: str) -> float:
    if not terms:
        return float("nan")
    h = normalize(hyp)
    return sum(normalize(t) in h for t in terms) / len(terms)

FILLERS = {"uh", "um", "umm", "uhh", "hmm", "mm", "ah", "ahh", "er", "erm"}


def normalize_verbatim(text: str) -> str:
    """Lowercase + strip punctuation and <tags>, but KEEP fillers (uh, um)."""
    text = re.sub(r"[<\[][^>\]]*[>\]]", " ", text)   # drop <inaudible>, <crosstalk>
    text = re.sub(r"[^a-z0-9' ]", " ", text.lower().replace("-", " "))
    return re.sub(r"\s+", " ", text).strip()


def wer_verbatim(ref: str, hyp: str) -> float:
    r, h = normalize_verbatim(ref), normalize_verbatim(hyp)
    return jiwer.wer(r, h) if r else float("nan")


def filler_count(text: str) -> int:
    return sum(w in FILLERS for w in normalize_verbatim(text).split())


def alignment_text(ref: str, hyp: str) -> str:
    """Word-by-word REF vs HYP with substitutions/deletions/insertions marked."""
    out = jiwer.process_words(normalize_verbatim(ref), normalize_verbatim(hyp))
    return jiwer.visualize_alignment(out)