from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class OrderRecord(BaseModel):
    """Internal representation; may contain private fields."""

    model_config = ConfigDict(extra="allow")


class SafeOrderResult(BaseModel):
    """Customer-safe order facts allowed beyond the service boundary."""

    model_config = ConfigDict(extra="forbid")

    order_id: str
    status: Optional[str] = None
    status_updated_at: Optional[datetime] = None

    membership_tier: Optional[str] = None
    placed_at: Optional[datetime] = None
    shipped_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None

    carrier: Optional[str] = None
    tracking_number: Optional[str] = None
    estimated_delivery: Optional[str] = None

    items: Optional[List[Dict[str, Any]]] = None
    customer_safe_message: Optional[str] = None
