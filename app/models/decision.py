from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class AgentIntent(str, Enum):
    POLICY_QUERY = "policy_query"
    ORDER_QUERY = "order_query"
    MIXED_QUERY = "mixed_query"
    ACTION_REQUEST = "action_request"
    PRIVACY_REQUEST = "privacy_request"
    SECURITY_REQUEST = "security_request"
    GENERAL = "general"
    UNSUPPORTED = "unsupported"


class HandoffReason(str, Enum):
    SOURCE_CONFLICT = "source_conflict"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    ORDER_NOT_FOUND = "order_not_found"
    ORDER_EXCEPTION = "order_exception"
    UNSUPPORTED_ACTION = "unsupported_action"
    PRIVACY_RESTRICTION = "privacy_restriction"
    SECURITY_RESTRICTION = "security_restriction"


class AgentDecision(BaseModel):
    intent: AgentIntent

    requires_rag: bool = False
    requires_order_lookup: bool = False

    order_id: Optional[str] = None

    requires_clarification: bool = False

    requires_handoff: bool = False
    handoff_reason: Optional[HandoffReason] = None

    evidence_ids: List[str] = Field(
        default_factory=list
    )