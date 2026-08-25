from app.config import settings
from app.orders.repository import OrderRepository
from app.orders.service import OrderService
from app.tools.order_lookup import OrderLookupTool


def create_order_lookup_tool() -> OrderLookupTool:
    repository = OrderRepository(
        settings.orders_path
    )

    service = OrderService(repository)

    return OrderLookupTool(service)