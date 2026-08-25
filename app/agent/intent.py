import re
from typing import Iterable, Optional


ORDER_ID_PATTERN = re.compile(
    r"\bORD-\d{4}\b",
    re.IGNORECASE,
)


POLICY_TERMS = {
    "policy",
    "return",
    "returns",
    "refund",
    "warranty",
    "exchange",
    "shipping",
    "ship",
    "membership",
    "trailplus",
    "eligible",
    "eligibility",
    "covered",
    "coverage",
    "final sale",
    "damaged",
    "damage",
    "broken",
    "wrong item",
    "international",
    "internationally",
    "canada",
    "canadian",
    "germany",
    "german",
    "destination",
    "destinations",
    "delivery window",
    "return window",
    "cancellation",
    "cancel",
    "price adjustment",
    "gift card",
}


KNOWLEDGE_TERMS = {
    "capacity",
    "material",
    "materials",
    "fabric",
    "fabrics",
    "adhesive",
    "adhesives",
    "vegan",
    "certification",
    "size",
    "weight",
    "dimension",
    "dimensions",
    "feature",
    "features",
    "specification",
    "specifications",
    "specs",
    "compatible",
    "compatibility",
    "cleaning",
    "care",
    "wash",
    "washed",
    "dishwasher",
    "microwave",
    "available",
    "price",
    "cost",
    "details",
    "information",
}


ORDER_PATTERNS = (
    r"\bord\s*-?\s*\d{4}\b",
    r"\bwhere\s+is\s+(?:my\s+)?(?:order|package|parcel|shipment)\b",
    r"\btrack(?:ing)?\b",
    r"\border\s+status\b",
    r"\bshipment\s+status\b",
    r"\bpackage\s+status\b",
    r"\bhas\s+(?:my\s+)?(?:order|package|parcel)\s+(?:shipped|arrived|been delivered)\b",
    r"\bwhen\s+(?:will|should|does|is)\b.*\b(?:arrive|get here|delivery|delivered)\b",
    r"\bdelivery\s+estimate\b",
    r"\bestimated\s+delivery\b",
    r"\bcarrier\b",
    r"\btracking\s+number\b",
)


ACTION_PATTERNS = (
    r"^\s*(?:please\s+)?(?:cancel|refund|replace|change|update|modify)\b",
    r"\b(?:can|could|would|will)\s+you\s+(?:cancel|refund|replace|change|update|modify)\b",
    r"\b(?:issue|give)\s+me\s+(?:a\s+)?refund\b",
    r"\bchange\s+(?:my|the)\s+(?:shipping\s+)?address\b",
)


SECURITY_PATTERNS = (
    r"\bsystem\s+prompt\b",
    r"\bhidden\s+(?:prompt|instruction|instructions)\b",
    r"\bdeveloper\s+(?:message|instructions?)\b",
    r"\bapi\s*key\b",
    r"\bsecret(?:s)?\b",
    r"\binternal\s+prompt\b",
)


SENSITIVE_ALWAYS_PATTERNS = (
    r"\brisk\s+score\b",
    r"\bfraud\s+(?:score|review|metadata)\b",
    r"\binternal\s+notes?\b",
    r"\bwarehouse\s+notes?\b",
    r"\bsupport\s+tags?\b",
    r"\binternal[-\s]+only\b",
)


SENSITIVE_PII_PATTERNS = (
    r"\bcustomer(?:'s)?\s+(?:email|address|phone(?:\s+number)?)\b",
    r"\b(?:give|show|reveal|provide|tell)\b.{0,80}\b(?:email|email address|home address|shipping address|phone number)\b",
    r"\b(?:email|address|phone number)\b.{0,60}\b(?:customer|recipient|buyer)\b",
)


FOLLOWUP_PATTERNS = (
    r"^\s*(?:and\s+)?what\s+about\b",
    r"^\s*how\s+about\b",
    r"^\s*and\s+how\s+long\b",
    r"^\s*how\s+long\s+does\s+it\s+take\b",
    r"^\s*what\s+if\b",
    r"^\s*does\s+that\b",
    r"^\s*is\s+that\b",
    r"^\s*and\s+(?:can|does|is|are|when|where|why|how)\b",
)


def extract_order_id(text: str) -> Optional[str]:
    match = ORDER_ID_PATTERN.search(text)
    if not match:
        return None
    return match.group(0).upper()


def _contains_term(text: str, term: str) -> bool:
    normalized = text.lower()
    if " " in term:
        return term in normalized
    return bool(re.search(rf"\b{re.escape(term)}\b", normalized))


def _contains_any_term(text: str, terms: Iterable[str]) -> bool:
    return any(_contains_term(text, term) for term in terms)


def has_order_intent(text: str) -> bool:
    normalized = text.lower()
    return any(re.search(pattern, normalized) for pattern in ORDER_PATTERNS)


def has_policy_intent(text: str) -> bool:
    return _contains_any_term(text, POLICY_TERMS)


def has_knowledge_intent(text: str) -> bool:
    return _contains_any_term(text, KNOWLEDGE_TERMS)


def has_sensitive_data_intent(text: str) -> bool:
    normalized = text.lower()

    # Address-change requests are actions/policy questions, not requests to
    # disclose a customer's stored address.
    if re.search(r"\b(?:change|update|modify)\s+(?:my|the)\s+(?:shipping\s+)?address\b", normalized):
        return False

    return any(
        re.search(pattern, normalized)
        for pattern in (*SENSITIVE_ALWAYS_PATTERNS, *SENSITIVE_PII_PATTERNS)
    )


def has_security_intent(text: str) -> bool:
    normalized = text.lower()
    return any(re.search(pattern, normalized) for pattern in SECURITY_PATTERNS)


def has_action_intent(text: str) -> bool:
    normalized = text.lower()
    return any(re.search(pattern, normalized) for pattern in ACTION_PATTERNS)


def is_contextual_followup(text: str) -> bool:
    normalized = text.lower()
    return any(re.search(pattern, normalized) for pattern in FOLLOWUP_PATTERNS)


def references_active_order(text: str) -> bool:
    normalized = text.lower()
    return bool(
        re.search(
            r"\b(?:it|this order|that order|the order|my order|this item|that item|this product|that product)\b",
            normalized,
        )
    )
