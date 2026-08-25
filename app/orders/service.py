import re

from app.orders.exceptions import (
    InvalidOrderIdError,
    OrderNotFoundError,
)
from app.orders.repository import OrderRepository
from app.orders.schemas import OrderSummary


ORDER_ID_PATTERN = re.compile(r"^ORD-\d{4}$")
TERMINAL_NO_ETA_STATUSES = {"cancelled", "returned"}


class OrderService:
    """Read-only customer-safe order lookup service."""

    SAFE_FIELDS = {
        "order_id",
        "membership_tier",
        "items",
        "placed_at",
        "status",
        "status_updated_at",
        "shipped_at",
        "delivered_at",
        "carrier",
        "tracking_number",
        "estimated_delivery",
        "customer_safe_message",
    }

    def __init__(
        self,
        repository: OrderRepository,
    ):
        self.repository = repository

    def get_order(
        self,
        order_id: str,
    ) -> OrderSummary:
        if not isinstance(order_id, str):
            raise InvalidOrderIdError(
                "Order ID must be a string."
            )

        normalized_id = order_id.strip().upper()

        if not ORDER_ID_PATTERN.fullmatch(normalized_id):
            raise InvalidOrderIdError(
                f"Invalid order ID format: {order_id}"
            )

        raw_order = self.repository.get_by_id(normalized_id)

        if raw_order is None:
            raise OrderNotFoundError(
                f"Order not found: {normalized_id}"
            )

        safe = {
            key: raw_order.get(key)
            for key in self.SAFE_FIELDS
        }

        # Stale ETA fields must never survive the service boundary for orders
        # that are no longer expected to arrive. A cancelled order should also
        # not expose historical carrier/tracking details as if a shipment were
        # still active.
        status = str(safe.get("status") or "").lower()
        if status in TERMINAL_NO_ETA_STATUSES:
            safe["estimated_delivery"] = None
        if status == "cancelled":
            safe["carrier"] = None
            safe["tracking_number"] = None
            safe["shipped_at"] = None

        return OrderSummary.model_validate(safe)
