from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    role: str
    content: str


class SessionState(BaseModel):
    session_id: str

    active_order_id: Optional[str] = None
    active_topic: Optional[str] = None

    entities: Dict[str, str] = Field(
        default_factory=dict
    )

    recent_turns: List[ConversationTurn] = Field(
        default_factory=list
    )

    last_action: Optional[str] = None

    pending_action: Optional[str] = None

    turn_count: int = 0