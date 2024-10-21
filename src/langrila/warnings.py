import functools
import warnings

warnings.simplefilter("default", DeprecationWarning)


def custom_formatwarning(message, category, filename, lineno, line=None):
    return f"[{category.__name__}]:{message}\n"


warnings.formatwarning = custom_formatwarning


def deprecated_argument(
    arg: str,
    move_to: str | None = None,
    removal: str | None = None,
    since: str | None = None,
    alternative: str | None = None,
    module_name: str | None = None,
    details: str | None = None,
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
                message = ""
                if move_to:
                    message += f"Argument '{arg}' will be moved to {move_to}. Please specify this argument in {move_to}."
                else:
                    if since:
                        message += f" {arg} in {func.__name__}() will be deprecated since {since}"

                        if removal:
                            message += f" and will be removed in version {removal}"
                    else:
                        message = (
                            f"Argument '{arg}' is deprecated and will be removed in future versions"
                        )

                    if module_name:
                        message += f" for {module_name}"

                if not message.endswith("."):
                    message += "."

                if alternative:
                    message += f" Use '{alternative}' instead."

                if details:
                    message += f" For more details, see: {details}."

                warnings.warn(message, DeprecationWarning, stacklevel=2)

            return func(*args, **kwargs)

        return wrapper

    return decorator
