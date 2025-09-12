
import json
import pickle
import re
import math
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple

# NLTK tokenizer (used in your notebook)
import nltk
try:
    # Make sure punkt is available at runtime; no-op if already present
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
from nltk.tokenize import word_tokenize


# ===== Functions ported (and slightly cleaned) from your notebook =====

def min_edit_distance(s1: str, s2: str) -> int:
    """Levenshtein distance (supports Vietnamese with diacritics)."""
    len_s1, len_s2 = len(s1), len(s2)
    dp = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]
    for i in range(len_s1 + 1):
        dp[i][0] = i
    for j in range(len_s2 + 1):
        dp[0][j] = j
    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,     # deletion
                dp[i][j-1] + 1,     # insertion
                dp[i-1][j-1] + cost # substitution
            )
    return dp[len_s1][len_s2]


def clean_punctuation(text: str) -> str:
    """Normalize common Vietnamese punctuation & spacing (pandas-free)."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    text = str(text)

    # 1) normalize sequences of dots to ellipsis "..."
    text = re.sub(r'\.{2,}', '...', text)

    # 2) numbers like "1 . 2" -> "1.2"; keep big-number commas "1,000"
    text = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', text)
    text = re.sub(r'(\d)\s*,\s*(\d)', r'\1,\2', text)

    # 3) normalize spacing around punctuation
    text = re.sub(r'\s*([,;:?!…])\s*', r'\1 ', text)   # ensure space after punctuation
    text = re.sub(r'\s*\.\.\.\s*', ' ... ', text)      # ellipsis surrounded by spaces

    # 4) quotes and Vietnamese-specific marks
    text = text.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'")
    text = text.replace('–', '-').replace('—', '-')

    # 5) parentheses spacing
    text = re.sub(r'(?<=\w)\(', ' (', text)
    text = re.sub(r'\)(?=\w)', ') ', text)

    # 6) collapse multi-spaces
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


def clean_text(text: str) -> str:
    # Replace URLs
    text = re.sub(r'https?://\S+|www\.\S+', '<url>', text)

    # Replace emojis with <emoji>
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002600-\U000026FF"
        u"\U00002700-\U000027BF"
        "]"
    )
    text = emoji_pattern.sub('<emoji>', text)

    # Replace dates
    text = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', '<date>', text)
    text = re.sub(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b', '<date>', text)

    # Replace time
    text = re.sub(r'\b\d{1,2}:\d{2}(:\d{2})?\b', '<time>', text)

    # Replace numbers
    text = re.sub(r'\d+([.,]\d+)?', '<num>', text)
    return text


def custom_tokenize(text: str) -> List[str]:
    # 1) find special tokens like <url>, <num>, <date> etc.
    special_tokens = list(set(re.findall(r'<[^<> ]+>', text)))
    token_map = {tok: f"___SPECIALTOKEN{i}___" for i, tok in enumerate(special_tokens)}

    # 2) replace with placeholders
    for orig, placeholder in token_map.items():
        text = text.replace(orig, placeholder)

    # 3) ensure placeholders are separated
    for placeholder in token_map.values():
        text = re.sub(rf'({re.escape(placeholder)})([^\W_]+)', r'\1 \2', text)
        text = re.sub(rf'([^\W_]+)({re.escape(placeholder)})', r'\1 \2', text)
        text = re.sub(rf'({re.escape(placeholder)})([/%°\-+=±≥<>\.])', r'\1 \2', text)
        text = re.sub(rf'([/%°\-+=±≥<>\.])({re.escape(placeholder)})', r'\1 \2', text)

    # 4) split separators into standalone tokens
    text = re.sub(r'([/\-+=±%°₫.<>\[\]…])', r' \1 ', text)
    text = re.sub(r'\.\.\.', ' ... ', text)
    text = re.sub(r'…', ' … ', text)

    # 5) tokenize
    tokens = word_tokenize(text)

    # 6) restore placeholders
    reverse = {v: k for k, v in token_map.items()}
    tokens = [reverse.get(t, t) for t in tokens]
    return tokens


def clean_text_with_mapping(text: str):
    replacements = {"<url>": [], "<emoji>": [], "<date>": [], "<time>": [], "<num>": []}
    patterns = [
        (r'https?://\S+|www\.\S+', "<url>"),
        ("["
         + u"\U0001F600-\U0001F64F"
         + u"\U0001F300-\U0001F5FF"
         + u"\U0001F680-\U0001F6FF"
         + u"\U0001F1E0-\U0001F1FF"
         + u"\u2600-\u26FF"
         + u"\u2700-\u27BF" + "]", "<emoji>"),
        (r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', "<date>"),
        (r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b', "<date>"),
        (r'\b\d{1,2}:\d{2}(?::\d{2})?\b', "<time>"),
        (r'\d+(?:[.,]\d+)?', "<num>"),
    ]

    replaced = [False] * len(text)
    new_text = ""
    matches = []

    for pattern, placeholder in patterns:
        for match in re.finditer(pattern, text):
            start, end = match.start(), match.end()
            if any(replaced[start:end]):
                continue
            matches.append((start, end, match.group(), placeholder))
            for i in range(start, end):
                replaced[i] = True

    matches.sort()
    last_pos = 0
    for start, end, original, placeholder in matches:
        new_text += text[last_pos:start]
        new_text += placeholder
        replacements[placeholder].append(original)
        last_pos = end
    new_text += text[last_pos:]
    return new_text, replacements


def restore_tokens(tokens: List[str], replacements: Dict[str, List[str]]) -> List[str]:
    counts = {k: 0 for k in replacements}
    restored = []
    for tok in tokens:
        if tok in replacements:
            idx = counts[tok]
            if idx < len(replacements[tok]):
                restored.append(replacements[tok][idx])
                counts[tok] += 1
            else:
                restored.append(tok)
        else:
            restored.append(tok)
    return restored


def get_emission_prob(observed_word: str, vocab: set, max_distance: int = 5) -> Dict[str, float]:
    target_distance = min(max(len(observed_word) - 1, 0), max_distance)
    candidates = []
    for word in vocab:
        dist = min_edit_distance(observed_word, word)
        if dist <= target_distance:
            candidates.append((word, dist))

    if not candidates:
        # fallback: allow up to max_distance
        for word in vocab:
            dist = min_edit_distance(observed_word, word)
            if dist <= max_distance:
                candidates.append((word, dist))
        if not candidates:
            return {}

    # convert distance -> prob (inverse with smoothing)
    probs = {}
    for word, dist in candidates:
        prob = 1.0 / (1 + dist)
        probs[word] = prob

    total = sum(probs.values())
    if total <= 0:
        return {}
    for word in probs:
        probs[word] /= total
    return probs


def viterbi(observed_tokens: List[str],
            vocab: set,
            ngrams: Dict[Tuple[str, str], Dict[str, int]],
            transition_prob: Dict[Tuple[str, str, str], float],
            context_totals: Dict[Tuple[str, str], int],
            alpha: float,
            epsilon: float = 1e-8) -> List[str]:
    observed = ['<start1>', '<start2>'] + observed_tokens + ['<end1>', '<end2>']
    T = len(observed)
    V = [{} for _ in range(T)]
    path = {}

    Vstart_ctx = ('<start1>', '<start2>')

    # t = 2 (first real token)
    emis_probs = get_emission_prob(observed[2], vocab)
    for y in emis_probs:
        count = ngrams.get(Vstart_ctx, {}).get(y, 0)
        total = context_totals.get(Vstart_ctx, 0)
        V[2][y] = math.log(max(transition_prob.get((Vstart_ctx[0], Vstart_ctx[1], y), 0.0), epsilon))
        V[2][y] += math.log(emis_probs[y] + epsilon)
        path[y] = ['<start1>', '<start2>', y]

    # DP
    for t in range(3, T):
        emis_probs = get_emission_prob(observed[t], vocab)
        newpath = {}
        for y in emis_probs:
            best_prev, max_prob = None, -float('inf')
            for y1 in V[t-1]:
                trigram = (observed[t-2], y1, y)
                prob_trans = transition_prob.get(trigram, None)
                if prob_trans is None:
                    # backoff using raw counts + alpha
                    context = (observed[t-2], y1)
                    count = ngrams.get(context, {}).get(y, 0)
                    total = context_totals.get(context, 0)
                    Vv = max(len(vocab), 1)
                    prob_trans = (count + alpha) / (total + alpha * Vv) if total or count else alpha / (alpha * Vv)

                prob = V[t-1][y1] + math.log(prob_trans + epsilon) + math.log(emis_probs[y] + epsilon)
                if prob > max_prob:
                    max_prob, best_prev = prob, y1

            if best_prev is not None:
                V[t][y] = max_prob
                newpath[y] = path[best_prev] + [y]

        if not newpath:
            raise ValueError(f"No valid path at step {t}. Check transition/emission coverage.")
        path = newpath

    n = T - 1
    prob, state = max((V[n][y], y) for y in V[n])
    return path[state]


def correct_text(input_text: str,
                 vocab: set,
                 ngrams: Dict[Tuple[str, str], Dict[str, int]],
                 transition_prob: Dict[Tuple[str, str, str], float],
                 context_totals: Dict[Tuple[str, str], int],
                 alpha: float) -> str:
    text = clean_punctuation(input_text.lower())
    cleaned, replacements = clean_text_with_mapping(text)
    tokens = custom_tokenize(cleaned)

    corrected_tokens = viterbi(tokens, vocab, ngrams, transition_prob, context_totals, alpha)

    # strip special tokens
    if corrected_tokens[:2] == ['<start1>', '<start2>']:
        corrected_tokens = corrected_tokens[2:]
    if corrected_tokens[-2:] == ['<end1>', '<end2>']:
        corrected_tokens = corrected_tokens[:-2]

    restored = restore_tokens(corrected_tokens, replacements)
    corrected_text = " ".join(restored)
    corrected_text = clean_punctuation(corrected_text)
    return corrected_text


# ===== Thin class wrapper for loading artifacts and exposing .correct() =====

class HMMSpellChecker:
    def __init__(self, vocab, ngrams, transition_prob, context_totals, alpha=1e-4):
        self.vocab = set(vocab)
        self.ngrams = ngrams
        self.transition_prob = transition_prob
        self.context_totals = context_totals
        self.alpha = float(alpha)

    @classmethod
    def from_artifacts(cls, path: str = "artifacts"):
        p = Path(path)
        with open(p / "vocab.json", "r", encoding="utf-8") as f:
            vocab = json.load(f)
        with open(p / "ngrams.pkl", "rb") as f:
            ngrams = pickle.load(f)
        with open(p / "transition_prob.pkl", "rb") as f:
            transition_prob = pickle.load(f)
        with open(p / "context_totals.pkl", "rb") as f:
            context_totals = pickle.load(f)
        # optional config
        alpha = 1e-4
        cfg_path = p / "config.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                alpha = float(cfg.get("viterbi_alpha", alpha))
        return cls(vocab, ngrams, transition_prob, context_totals, alpha)

    def correct(self, text: str) -> str:
        return correct_text(text, self.vocab, self.ngrams, self.transition_prob, self.context_totals, self.alpha)
