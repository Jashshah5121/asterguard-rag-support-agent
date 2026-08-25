class OrderError(Exception):
    """Base class for order-related errors."""


class InvalidOrderIdError(OrderError):
    """Raised when an order ID has an invalid format."""


class OrderNotFoundError(OrderError):
    """Raised when a valid order ID does not exist."""