"""Exception hierarchy for `tracerax`.

Every error raised by this package derives from
[`TraceraxError`][tracerax.TraceraxError], so a caller can trap the whole
package with a single ``except`` clause. Each concrete error also inherits from
the closest standard-library exception, so code written against the stdlib
keeps working without importing anything from this package.
"""

from __future__ import annotations


__all__ = [
    "TraceraxError",
    "ValidationError",
]


class TraceraxError(Exception):
    """Base class for every exception raised by ``tracerax``.

    Examples:
        >>> from tracerax import TraceraxError, ValidationError
        >>> issubclass(ValidationError, TraceraxError)
        True
    """


class ValidationError(TraceraxError, ValueError):
    """Raised when an argument fails validation.

    Also a `ValueError`, so ``except ValueError`` still catches it.

    Examples:
        >>> from tracerax import ValidationError
        >>> issubclass(ValidationError, ValueError)
        True
    """
