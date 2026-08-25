import pytest

from app.config import settings
from app.orders.repository import OrderRepository
from app.orders.service import OrderService
from app.tools.order_lookup import OrderLookupTool


@pytest.fixture
def order_tool():
    repository = OrderRepository(
        settings.orders_path
    )

    service = OrderService(repository)

    return OrderLookupTool(service)


def test_valid_order_returns_structured_data(
    order_tool,
):
    result = order_tool.execute(
        "ORD-1003"
    )

    assert result["success"] is True

    data = result["data"]

    assert data["order_id"] == "ORD-1003"
    assert data["status"] == "shipped"
    assert data["carrier"] == "USPS"
    assert (
        data["tracking_number"]
        == "94001118995600001003"
    )


def test_tool_normalizes_order_id(
    order_tool,
):
    result = order_tool.execute(
        "  ord-1003 "
    )

    assert result["success"] is True
    assert result["data"]["order_id"] == "ORD-1003"


def test_invalid_order_id_returns_structured_error(
    order_tool,
):
    result = order_tool.execute(
        "1003"
    )

    assert result["success"] is False

    assert (
        result["error"]["code"]
        == "INVALID_ORDER_ID"
    )


def test_unknown_order_returns_structured_error(
    order_tool,
):
    result = order_tool.execute(
        "ORD-9999"
    )

    assert result["success"] is False

    assert (
        result["error"]["code"]
        == "ORDER_NOT_FOUND"
    )


def test_tool_never_exposes_internal_data(
    order_tool,
):
    result = order_tool.execute(
        "ORD-1005"
    )

    assert result["success"] is True

    data = result["data"]

    assert "internal" not in data
    assert "risk_score" not in str(data)
    assert "warehouse_note" not in str(data)
    assert "support_tags" not in str(data)


def test_tool_does_not_expose_adversarial_instruction(
    order_tool,
):
    result = order_tool.execute(
        "ORD-1005"
    )

    assert result["success"] is True

    serialized = str(
        result["data"]
    ).lower()

    assert "issue a $100 coupon" not in serialized
    assert "hide the delay reason" not in serialized


def test_fraud_metadata_is_not_exposed(
    order_tool,
):
    result = order_tool.execute(
        "ORD-1007"
    )

    assert result["success"] is True

    serialized = str(
        result["data"]
    ).lower()

    assert "risk_score" not in serialized
    assert "support_tags" not in serialized


def test_tool_is_read_only(
    order_tool,
):
    first = order_tool.execute(
        "ORD-1003"
    )

    second = order_tool.execute(
        "ORD-1003"
    )

    assert first == second