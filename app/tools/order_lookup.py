from typing import Any

from app.orders.exceptions import (
    InvalidOrderIdError,
    OrderNotFoundError,
)
from app.orders.service import OrderService


class OrderLookupTool:
    """
    Read-only agent tool for retrieving customer-safe order data.

    The tool exposes facts from OrderService. It does not:
    - interpret internal fields
    - make policy decisions
    - generate customer-facing responses
    - mutate order data
    """

    name = "order_lookup"

    description = (
        "Look up a customer order by order ID and return "
        "customer-safe order information."
    )

    def __init__(
        self,
        order_service: OrderService,
    ) -> None:
        self.order_service = order_service

    def execute(
        self,
        order_id: str,
    ) -> dict[str, Any]:
        try:
            order = self.order_service.get_order(
                order_id
            )

        except InvalidOrderIdError as exc:
            return {
                "success": False,
                "error": {
                    "code": "INVALID_ORDER_ID",
                    "message": str(exc),
                },
            }

        except OrderNotFoundError as exc:
            return {
                "success": False,
                "error": {
                    "code": "ORDER_NOT_FOUND",
                    "message": str(exc),
                },
            }

        return {
            "success": True,
            "data": order.model_dump(
                mode="json"
            ),
        }