"""What importing the package loads: the layer rule, as a test.

Each check runs in a fresh interpreter, because this one has already imported
whatever the other tests needed.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest


#: The stack libraries tracerax may load (AGENTS.md, "Boundaries"): functax and
#: integrix, and gaussx, which is integrix's and is never imported directly.
ALLOWED = {"functax", "gaussx", "integrix"}

#: Every import package of the stack, so that a new one is noticed by name.
STACK = ALLOWED | {
    "argminax",
    "diagnostix",
    "filterax",
    "geocatalog",
    "geocloud",
    "geonnax",
    "geoops",
    "geopatcher",
    "geophysix",
    "geoproducts",
    "geotoolz",
    "gridax",
    "kernellib",
    "pipekit",
    "pipekit_array",
    "pipekit_cycle",
    "pipekit_jax",
    "pipekit_ops",
    "plumax",
    "pyrox",
    "radiomax",
    "radtrax",
    "sesmax",
    "somax",
    "spectraldiffx",
    "vardax",
    "xr_toolz",
    "xrops",
    "xtremax",
}

_IMPORT_EVERYTHING = """
import importlib, json, pkgutil, sys
import tracerax
for info in pkgutil.walk_packages(tracerax.__path__, "tracerax."):
    importlib.import_module(info.name)
print(json.dumps(sorted({name.split(".")[0] for name in sys.modules})))
"""


def _loaded_by_importing_everything(first: str = "") -> set[str]:
    """The top-level packages loaded once every module of the package is.

    Args:
        first: Code run before the import, to check that a load is noticed.
    """
    run = subprocess.run(
        [sys.executable, "-c", first + _IMPORT_EVERYTHING],
        check=True,
        capture_output=True,
        text=True,
    )
    return set(json.loads(run.stdout))


def test_importing_the_package_loads_only_the_allowed_stack_libraries() -> None:
    assert _loaded_by_importing_everything() & STACK <= ALLOWED


def test_the_required_stack_libraries_load_only_the_allowed_ones() -> None:
    """What functax and integrix bring is allowed too, so a pin move is noticed."""
    loaded = _loaded_by_importing_everything("import functax, integrix.march\n")
    assert loaded & STACK <= ALLOWED
    # the check sees a stack library when one is loaded
    assert {"functax", "integrix"} <= loaded


@pytest.mark.gridax
def test_the_extra_is_not_loaded_by_the_package() -> None:
    assert "gridax" in _loaded_by_importing_everything("import gridax\n")
    assert "gridax" not in _loaded_by_importing_everything()
