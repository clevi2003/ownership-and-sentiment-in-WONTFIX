import math
from nltk.sentiment import SentimentIntensityAnalyzer
from utils.regex_expressions import TOKEN_PATTERN, PATHISH_PATTERN_SENTIMENT, URL_PATTERN, ALPHA_CHAR_PATTERN, UPPERCASE_CHAR_PATTERN
from vadersentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from utils.io_helpers import clean_text


class FallbackLexiconSentimentAnalyzer:
    POSITIVE_WORDS = {
        "good", "great", "excellent", "amazing", "awesome", "nice", "helpful", "thanks", "thank",
        "love", "loved", "like", "liked", "success", "successful", "fixed", "fix", "works", "working",
        "resolved", "resolve", "clear", "clean", "improved", "improve", "correct", "stable", "fast",
    }
    NEGATIVE_WORDS = {
        "bad", "terrible", "awful", "hate", "broken", "break", "breaking", "bug", "bugs", "error",
        "errors", "fail", "failed", "failing", "failure", "regression", "worse", "worst", "problem",
        "problems", "issue", "issues", "confusing", "unclear", "angry", "annoying", "frustrating",
        "slow", "blocked", "blocker", "wrong", "crash", "crashes", "crashed",
    }
    INTENSIFIERS = {"very", "really", "extremely", "highly", "super", "too"}
    NEGATORS = {"not", "no", "never", "none", "without", "hardly", "barely"}

    def polarity_scores(self, text):
        text = clean_text(text) or ""
        if not text:
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

        tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
        if not tokens:
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

        pos_score = 0.0
        neg_score = 0.0
        for index, token in enumerate(tokens):
            weight = 1.0
            if index > 0 and tokens[index - 1] in self.INTENSIFIERS:
                weight *= 1.5
            is_negated = index > 0 and tokens[index - 1] in self.NEGATORS

            if token in self.POSITIVE_WORDS:
                if is_negated:
                    neg_score += weight
                else:
                    pos_score += weight
            elif token in self.NEGATIVE_WORDS:
                if is_negated:
                    pos_score += weight
                else:
                    neg_score += weight

        total = pos_score + neg_score
        if total <= 0:
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

        pos = pos_score / total
        neg = neg_score / total
        compound = (pos_score - neg_score) / (total + 1.0)
        compound = max(min(compound, 1.0), -1.0)
        neu = max(0.0, 1.0 - min(1.0, pos + neg))
        return {"neg": neg, "neu": neu, "pos": pos, "compound": compound}


class SentimentAnalyzerWrapper:
    def __init__(self):
        self.backend_name = None
        self._analyzer = self._build_analyzer()

    def _build_analyzer(self):
        try:
            try:
                analyzer = SentimentIntensityAnalyzer()
            except LookupError:
                import nltk
                nltk.download("vader_lexicon", quiet=True)
                analyzer = SentimentIntensityAnalyzer()
            self.backend_name = "nltk_vader"
            return analyzer
        except Exception:
            pass

        try:
            self.backend_name = "vaderSentiment"
            return SentimentIntensityAnalyzer()
        except Exception:
            pass

        self.backend_name = "fallback_lexicon"
        return FallbackLexiconSentimentAnalyzer()

    def polarity_scores(self, text):
        return self._analyzer.polarity_scores(text or "")


def score_text_features(text, analyzer):
    clean = clean_text(text)
    if not clean:
        return {
            "text": None,
            "has_text": 0,
            "text_length_chars": 0,
            "token_count": 0,
            "question_mark_count": 0,
            "exclamation_mark_count": 0,
            "uppercase_ratio": 0.0,
            "has_code_block": 0,
            "has_url": 0,
            "has_path_reference": 0,
            "sentiment_neg": 0.0,
            "sentiment_neu": 1.0,
            "sentiment_pos": 0.0,
            "sentiment_compound": 0.0,
        }

    scores = analyzer.polarity_scores(clean)
    alpha_chars = ALPHA_CHAR_PATTERN.findall(clean)
    alpha_count = len(alpha_chars)
    uppercase_count = len(UPPERCASE_CHAR_PATTERN.findall(clean))
    tokens = TOKEN_PATTERN.findall(clean)

    return {
        "text": clean,
        "has_text": 1,
        "text_length_chars": len(clean),
        "token_count": len(tokens),
        "question_mark_count": clean.count("?"),
        "exclamation_mark_count": clean.count("!"),
        "uppercase_ratio": float(uppercase_count) / float(alpha_count) if alpha_count else 0.0,
        "has_code_block": 1 if "```" in clean or "`" in clean else 0,
        "has_url": 1 if URL_PATTERN.search(clean) else 0,
        "has_path_reference": 1 if PATHISH_PATTERN_SENTIMENT.search(clean) else 0,
        "sentiment_neg": float(scores.get("neg", 0.0)),
        "sentiment_neu": float(scores.get("neu", 1.0)),
        "sentiment_pos": float(scores.get("pos", 0.0)),
        "sentiment_compound": float(scores.get("compound", 0.0)),
    }


def safe_divide(numerator, denominator, default_value=0.0):
    if denominator in {0, 0.0, None}:
        return default_value
    try:
        return float(numerator) / float(denominator)
    except Exception:
        return default_value


def compute_series_slope(values):
    numeric_values = [float(value) for value in values if value is not None and not pd.isna(value)]
    if len(numeric_values) < 2:
        return 0.0

    count = len(numeric_values)
    x_mean = (count - 1) / 2.0
    y_mean = sum(numeric_values) / float(count)

    numerator = 0.0
    denominator = 0.0
    for index, value in enumerate(numeric_values):
        x_delta = float(index) - x_mean
        y_delta = float(value) - y_mean
        numerator += x_delta * y_delta
        denominator += x_delta * x_delta

    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def take_mean(values):
    numeric_values = [float(value) for value in values if value is not None and not pd.isna(value)]
    if not numeric_values:
        return 0.0
    return float(sum(numeric_values)) / float(len(numeric_values))


def take_std(values):
    numeric_values = [float(value) for value in values if value is not None and not pd.isna(value)]
    if len(numeric_values) < 2:
        return 0.0
    mean_value = take_mean(numeric_values)
    variance = sum((value - mean_value) ** 2 for value in numeric_values) / float(len(numeric_values) - 1)
    return math.sqrt(max(variance, 0.0))


def take_median(values):
    numeric_values = sorted(float(value) for value in values if value is not None and not pd.isna(value))
    if not numeric_values:
        return 0.0
    count = len(numeric_values)
    midpoint = count // 2
    if count % 2 == 1:
        return numeric_values[midpoint]
    return (numeric_values[midpoint - 1] + numeric_values[midpoint]) / 2.0


def split_early_late(values):
    cleaned = [value for value in values if value is not None and not pd.isna(value)]
    if not cleaned:
        return [], []
    if len(cleaned) == 1:
        return cleaned, cleaned
    midpoint = max(1, len(cleaned) // 2)
    early = cleaned[:midpoint]
    late = cleaned[midpoint:]
    if not late:
        late = cleaned[-1:]
    return early, late
