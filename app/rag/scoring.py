import re
from typing import Set


STOPWORDS: Set[str] = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "the",
    "to",
    "under",
    "what",
    "when",
    "where",
    "with",
}


INTENT_PATTERNS = {
    "duration": (
        r"\bhow long\b",
        r"\bhow many days\b",
        r"\bwithin how many days\b",
        r"\bhow much time\b",
        r"\btime limit\b",
        r"\breturn window\b",
        r"\bwindow\b",
        r"\bdeadline\b",
    ),
    "coverage": (
        r"\bwhat does .* cover\b",
        r"\bwhat is covered\b",
        r"\bwhat.*covered\b",
        r"\bcoverage\b",
    ),
    "eligibility": (
        r"\bcan i\b",
        r"\bam i eligible\b",
        r"\beligible\b",
        r"\bwho can\b",
    ),
    "shipping_destination": (
        r"\bdo you ship\b",
        r"\bship to\b",
        r"\bshipping to\b",
        r"\bdeliver to\b",
        r"\bavailable in\b",
    ),
}


INTENT_TERMS = {
    "duration": {
        "window",
        "period",
        "deadline",
        "estimate",
        "days",
        "time",
    },
    "coverage": {
        "covered",
        "coverage",
        "includes",
        "included",
        "warranty",
    },
    "eligibility": {
        "eligible",
        "eligibility",
        "qualify",
    },
    "shipping_destination": {
        "destination",
        "shipping",
        "delivery",
        "international",
    },
}


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )
        if token not in STOPWORDS
    ]


def overlap_score(
    query: str,
    text: str,
) -> float:
    query_tokens = set(
        tokenize(query)
    )

    text_tokens = set(
        tokenize(text)
    )

    if not query_tokens:
        return 0.0

    return len(
        query_tokens & text_tokens
    ) / len(query_tokens)


def detect_intents(
    query: str,
) -> set[str]:
    normalized = query.lower()

    detected = set()

    for intent, patterns in INTENT_PATTERNS.items():
        if any(
            re.search(pattern, normalized)
            for pattern in patterns
        ):
            detected.add(intent)

    return detected


def normalize_intent_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"

    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]

    return token


def intent_heading_score(
    query: str,
    heading: str,
) -> float:
    intents = detect_intents(query)

    if not intents:
        return 0.0

    heading_tokens = {
        normalize_intent_token(token)
        for token in tokenize(heading)
    }

    intent_weights = {
        "shipping_destination": {
            "international": 1.0,
            "destination": 1.0,
            "shipping": 0.4,
            "delivery": 0.3,
        },
        "duration": {
            "window": 1.0,
            "period": 0.8,
            "days": 0.8,
            "estimate": 0.7,
            "deadline": 0.7,
            "time": 0.5,
        },
        "coverage": {
            "warranty": 1.0,
            "coverage": 0.9,
            "covered": 0.7,
            "included": 0.6,
            "includes": 0.6,
        },
        "eligibility": {
            "eligible": 1.0,
            "eligibility": 1.0,
            "qualify": 0.9,
        },
    }

    scores = []

    for intent in intents:
        weights = intent_weights.get(
            intent,
            {},
        )

        if not weights:
            continue

        total_weight = sum(
            weights.values()
        )

        matched_weight = sum(
            weight
            for term, weight in weights.items()
            if normalize_intent_token(term) in heading_tokens
        )

        scores.append(
            matched_weight / total_weight
        )

    return max(
        scores,
        default=0.0,
    )