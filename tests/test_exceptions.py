"""Tests for the exception hierarchy.

The hierarchy is part of the public contract: downstream code catches these,
so a change in the inheritance graph is a breaking change.
"""

from __future__ import annotations

import pytest

from tracerax import TraceraxError, ValidationError


SUBCLASSES = [ValidationError]


@pytest.mark.parametrize("error", SUBCLASSES)
def test_every_error_derives_from_the_base(error: type[Exception]) -> None:
    assert issubclass(error, TraceraxError)


def test_the_base_is_not_a_value_error() -> None:
    """A future error may be a `TypeError`, so only the subclasses say which."""
    assert not issubclass(TraceraxError, ValueError)


@pytest.mark.parametrize("error", SUBCLASSES)
def test_subclasses_are_caught_as_value_errors(error: type[Exception]) -> None:
    with pytest.raises(ValueError):
        raise error("boom")


def test_a_single_except_clause_catches_the_package() -> None:
    caught: list[TraceraxError] = []
    for error in SUBCLASSES:
        try:
            raise error("boom")
        except TraceraxError as err:
            caught.append(err)
    assert [type(err) for err in caught] == SUBCLASSES


def test_subclasses_have_a_docstring() -> None:
    assert all(error.__doc__ for error in SUBCLASSES)
