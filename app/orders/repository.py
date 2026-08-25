import json
from pathlib import Path
from typing import Any


class OrderRepository:
    """
    Read-only repository over the supplied order snapshot.

    This layer returns raw records. It does not expose them directly
    to customers; the service layer performs the public projection.
    """

    def __init__(self, orders_path: Path):
        self.orders_path = orders_path
        self._orders: dict[str, dict[str, Any]] | None = None

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._orders is None:
            payload = json.loads(
                self.orders_path.read_text(
                    encoding="utf-8"
                )
            )

            orders = payload.get("orders")

            if not isinstance(orders, list):
                raise ValueError(
                    "Order dataset must contain an 'orders' list."
                )

            self._orders = {
                order["order_id"]: order
                for order in orders
            }

        return self._orders

    def get_by_id(
        self,
        order_id: str,
    ) -> dict[str, Any] | None:
        orders = self._load()

        return orders.get(order_id)