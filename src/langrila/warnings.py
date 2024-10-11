import functools
import logging
import warnings


def custom_formatwarning(message, category, filename, lineno, line=None):
    return f"{category.__name__}: {message}\n"


LOGGER = logging.getLogger(__name__)


def deprecated_argument(
    arg: str, move_to: str | None = None, removal: str | None = None, since: str | None = None
):
    """
    Decorator to mark arguments as obsolete.

    Args:
        arg_name (str): Obsolete Argument Name.
        new_func_name (str): New function name.
        version (str, Optional): Version information that will be discontinued.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if arg in kwargs and kwargs[arg]:
                if move_to:
                    message = f"Argument '{arg}' will be moved to {move_to}. Please specify this argument in {move_to}."
                else:
                    message = (
                        f"Argument '{arg}' is deprecated and will be removed in future versions."
                    )

                if since:
                    message += f" {arg} in {func.__name__} will be deprecated since {since}"

                    if removal:
                        message += f" and will be removed in version {removal}."

                if not message.endswith("."):
                    message += "."

                LOGGER.warning(message)
            return func(*args, **kwargs)

        return wrapper

    return decorator
