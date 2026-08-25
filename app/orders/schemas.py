from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class OrderItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku: str
    name: str
    quantity: int
    final_sale: bool


class OrderSummary(BaseModel):
    """Explicit customer-safe projection of an order record.

    Customer contact details and internal metadata do not exist in this schema,
    so they cannot accidentally leak through a future model_dump/tool call.
    """

    model_config = ConfigDict(extra="forbid")

    order_id: str
    membership_tier: str
    items: list[OrderItem]

    placed_at: datetime
    status: str
    status_updated_at: datetime

    shipped_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None

    carrier: Optional[str] = None
    tracking_number: Optional[str] = None
    estimated_delivery: Optional[str] = None

    customer_safe_message: str
