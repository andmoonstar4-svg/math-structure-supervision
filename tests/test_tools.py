import pytest

from math_structure.tools import (
    TOOL_REGISTRY,
    ToolArgumentError,
    execute_tool,
    tool_args_semantically_equivalent,
)


@pytest.mark.parametrize(
    ("tool", "tool_args", "answer"),
    [
        (
            "elimination",
            {"equations": ["2*x + y - 7", "x - y - 2"], "target": "x"},
            "3",
        ),
        (
            "substitution",
            {
                "mode": "solve_system",
                "equations": ["y - x - 1", "x + y - 5"],
                "target": "y",
            },
            "3",
        ),
        (
            "substitution",
            {
                "mode": "evaluate",
                "expression": "2*t**2 - t + 1",
                "substitutions": {"t": 3},
            },
            "16",
        ),
        (
            "factorization",
            {
                "mode": "evaluate",
                "expression": "u**2 - v**2",
                "substitutions": {"u": 9, "v": 4},
            },
            "65",
        ),
        (
            "factorization",
            {
                "mode": "root_power_sum",
                "polynomial": "t**2 - 5*t + 6",
                "variable": "t",
                "power": 2,
            },
            "13",
        ),
        (
            "expansion",
            {
                "mode": "evaluate",
                "expression": "(x + 2)*(x - 1)",
                "substitutions": {"x": 4},
            },
            "18",
        ),
        (
            "expansion",
            {"mode": "symmetric_square_sum", "sum": 5, "product": 6},
            "13",
        ),
        (
            "proportional_scaling",
            {"base_quantity": 3, "base_value": 8, "target_quantity": 12},
            "32",
        ),
        (
            "cross_multiplication",
            {"left_num": 3, "left_den": 5, "right_num": 12},
            "20",
        ),
        ("iteration", {"start": -2, "step": 4, "updates": 5}, "18"),
    ],
)
def test_each_plugin_and_mode(tool: str, tool_args: dict, answer: str) -> None:
    assert execute_tool(tool, tool_args) == answer


def test_registry_matches_taxonomy_v2_tools() -> None:
    assert set(TOOL_REGISTRY) == {
        "substitution",
        "elimination",
        "factorization",
        "expansion",
        "proportional_scaling",
        "cross_multiplication",
        "iteration",
    }
    assert execute_tool("iteration", {"start": 1, "step": 2, "updates": 3}) == "7"


def test_plugins_fail_explicitly_on_bad_arguments() -> None:
    with pytest.raises(ToolArgumentError, match="unknown tool"):
        execute_tool("guess", {})
    with pytest.raises(ToolArgumentError, match="missing tool arguments"):
        execute_tool("iteration", {"start": 1})
    with pytest.raises(ToolArgumentError, match="cannot be zero"):
        execute_tool(
            "proportional_scaling",
            {"base_quantity": 0, "base_value": 2, "target_quantity": 4},
        )


def test_semantic_argument_equivalence_accepts_symbolic_rewrites() -> None:
    gold = {
        "equations": ["2*x + y - 7", "x - y - 2"],
        "target": "x",
    }
    equivalent = {
        "equations": ["2*x - 2*y - 4", "y + 2*x - 7"],
        "target": "x",
    }
    assert tool_args_semantically_equivalent("elimination", equivalent, gold)

    expansion_gold = {
        "mode": "evaluate",
        "expression": "(x + 2)*(x - 1)",
        "substitutions": {"x": 4},
    }
    expansion_equivalent = {
        "mode": "evaluate",
        "expression": "x**2 + x - 2",
        "substitutions": {"x": 4},
    }
    assert tool_args_semantically_equivalent(
        "expansion", expansion_equivalent, expansion_gold
    )
    assert not tool_args_semantically_equivalent(
        "expansion",
        {**expansion_equivalent, "substitutions": {"x": 5}},
        expansion_gold,
    )
    assert not tool_args_semantically_equivalent(
        "expansion",
        {**expansion_equivalent, "expression": "2*(x**2 + x - 2)"},
        expansion_gold,
    )
