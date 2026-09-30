# tests/proof_of_boundary/test_pb7_hitl_interrupt_propagation.py
#
# PB-7 (HITL interrupt propagation) applies only to templates whose nodes call
# interrupt(). This template has HITL disabled, so the boundary it must hold is
# the inverse: config/config.yaml and the node code have to agree.
#
#   - hitl.enabled is not true  -> no node may call interrupt() (an interrupt
#     without a checkpointer silently no-ops or deadlocks the run).
#   - hitl.enabled is true      -> at least one node must call interrupt(), and
#     the interrupt()/GraphInterrupt propagation tests for that node must be
#     added to this file before release.

from __future__ import annotations

import ast
import pathlib

import yaml

_ROOT = pathlib.Path(__file__).parents[2]
_CONFIG_PATH = _ROOT / "config" / "config.yaml"
_NODES_DIR = _ROOT / "src" / "nodes"


def _hitl_enabled() -> bool:
    """Return True when config/config.yaml declares hitl.enabled: true."""
    data = yaml.safe_load(_CONFIG_PATH.read_text()) or {}
    hitl = data.get("hitl", {})
    assert isinstance(hitl, dict), "config/config.yaml: 'hitl' must be a mapping"
    return bool(hitl.get("enabled", False))


def _nodes_calling_interrupt() -> list[str]:
    """Names of node modules that call interrupt(...) (bare or attribute form)."""
    callers = []
    for path in sorted(_NODES_DIR.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name == "interrupt":
                callers.append(path.name)
                break
    return callers


def test_config_yaml_exists_and_declares_hitl_mapping_shape() -> None:
    assert _CONFIG_PATH.exists(), "config/config.yaml must ship so PB-7 can verify hitl.enabled"
    _hitl_enabled()


def test_pb7_hitl_config_matches_interrupt_usage() -> None:
    """hitl.enabled and interrupt() usage in src/nodes/ must agree."""
    callers = _nodes_calling_interrupt()
    if _hitl_enabled():
        assert callers, (
            "hitl.enabled is true but no node calls interrupt(); add the D6 interrupt() pattern "
            "and its PB-7 propagation tests, or disable HITL in config/config.yaml"
        )
    else:
        assert callers == [], f"HITL is disabled but these nodes call interrupt(): {callers}"
