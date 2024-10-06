import functools
import warnings


def custom_formatwarning(message, category, filename, lineno, line=None):
    return f"{category.__name__}: {message}\n"


warnings.simplefilter("always", DeprecationWarning)
warnings.formatwarning = custom_formatwarning


def deprecated_argument(
    old_arg_name: str, new_arg_name: str | None = None, version: str | None = None
):
    """
    Decorator to mark arguments as obsolete.

    Args:
        old_arg_name (str): Obsolete Argument Name.
        new_arg_name (str): New argument name.
        version (str, Optional): Version information that will be discontinued.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if old_arg_name in kwargs:
                message = f"Argument '{old_arg_name}' will be deprecated."
                if new_arg_name:
                    message += f" Use '{new_arg_name}' instead."
                if version:
                    message += f" This change will be introduced in version {version}."
                warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def change_function(arg_name: str, new_func_name: str, version: str | None = None):
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
            if arg_name in kwargs:
                message = f"Argument '{arg_name}' will be moved to {new_func_name}. Please specify this argument in {new_func_name}."

                if version:
                    message += f" This change will be introduced in version {version}."
                warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def deprecated(reason: str | None = None, version: str | None = None):
    """
    A decorator that marks a function or method as obsolete.

    Args:
        reason (str, Optional): Reasons for discontinuance and alternatives.
        version (str, Optional): Version information that will be discontinued.
    """

    def decorator(func):
        message = f"'{func.__name__}' will be deprecated."
        if reason:
            message += f" Reason: {reason}"
        if version:
            message += f" This change will be introduced in version {version}."

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator
