import pytest

from app.config import settings
from app.orders.exceptions import (
    InvalidOrderIdError,
    OrderNotFoundError,
)
from app.orders.repository import OrderRepository
from app.orders.service import OrderService


@pytest.fixture
def order_service():
    repository = OrderRepository(
        settings.orders_path
    )

    return OrderService(repository)


def test_existing_order_is_returned(order_service):
    order = order_service.get_order(
        "ORD-1003"
    )

    assert order.order_id == "ORD-1003"
    assert order.status == "shipped"
    assert order.carrier == "USPS"
    assert order.tracking_number == (
        "94001118995600001003"
    )


def test_order_lookup_is_case_insensitive(order_service):
    order = order_service.get_order(
        "ord-1003"
    )

    assert order.order_id == "ORD-1003"


def test_order_lookup_strips_whitespace(order_service):
    order = order_service.get_order(
        "  ORD-1003  "
    )

    assert order.order_id == "ORD-1003"


def test_invalid_order_id_is_rejected(order_service):
    with pytest.raises(
        InvalidOrderIdError
    ):
        order_service.get_order(
            "1003"
        )


def test_unknown_order_is_rejected(order_service):
    with pytest.raises(
        OrderNotFoundError
    ):
        order_service.get_order(
            "ORD-9999"
        )


def test_internal_fields_are_not_exposed(order_service):
    order = order_service.get_order(
        "ORD-1005"
    )

    dumped = order.model_dump()

    assert "internal" not in dumped
    assert "risk_score" not in dumped
    assert "warehouse_note" not in dumped
    assert "support_tags" not in dumped


def test_adversarial_internal_instruction_is_not_exposed(
    order_service,
):
    order = order_service.get_order(
        "ORD-1005"
    )

    dumped = order.model_dump()

    serialized = str(dumped).lower()

    assert "issue a $100 coupon" not in serialized
    assert "hide the delay reason" not in serialized


def test_fraud_review_details_are_not_exposed(
    order_service,
):
    order = order_service.get_order(
        "ORD-1007"
    )

    dumped = order.model_dump()

    assert "internal" not in dumped
    assert "risk_score" not in dumped
    assert "warehouse_note" not in dumped


def test_customer_safe_message_is_preserved(
    order_service,
):
    order = order_service.get_order(
        "ORD-1005"
    )

    assert (
        order.customer_safe_message
        == (
            "The carrier reported a weather delay. "
            "The current estimated delivery date is "
            "August 20, 2026."
        )
    )