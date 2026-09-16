"""Deterministic symbolic tools for the taxonomy-v2 hybrid experiment."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import sympy as sp


class PluginError(ValueError):
    """Base class for explicit deterministic-plugin failures."""


class ToolArgumentError(PluginError):
    """The supplied arguments do not match the selected tool schema."""


class ToolExecutionError(PluginError):
    """The arguments are valid, but deterministic execution cannot finish."""


def _require_mapping(args: Any) -> dict[str, Any]:
    if not isinstance(args, Mapping):
        raise ToolArgumentError("tool_args must be a JSON object")
    return dict(args)


def _require_keys(
    args: Any,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    values = _require_mapping(args)
    optional = optional or set()
    missing = required - values.keys()
    extra = values.keys() - required - optional
    if missing:
        raise ToolArgumentError(f"missing tool arguments: {sorted(missing)}")
    if extra:
        raise ToolArgumentError(f"unexpected tool arguments: {sorted(extra)}")
    return values


def _expression(value: Any, field: str) -> sp.Expr:
    if not isinstance(value, str) or not value.strip():
        raise ToolArgumentError(f"{field} must be a non-empty expression string")
    try:
        return sp.sympify(value)
    except (sp.SympifyError, TypeError, ValueError) as exc:
        raise ToolArgumentError(f"invalid symbolic expression in {field}") from exc


def _number(value: Any, field: str) -> sp.Expr:
    try:
        result = sp.sympify(value)
    except (sp.SympifyError, TypeError, ValueError) as exc:
        raise ToolArgumentError(f"{field} must be numeric") from exc
    if result.free_symbols or result.is_number is not True:
        raise ToolArgumentError(f"{field} must be numeric")
    return result


def _substitutions(value: Any) -> dict[sp.Symbol, sp.Expr]:
    mapping = _require_mapping(value)
    substitutions: dict[sp.Symbol, sp.Expr] = {}
    for name, number in mapping.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ToolArgumentError("substitution keys must be variable names")
        substitutions[sp.Symbol(name)] = _number(number, f"substitutions.{name}")
    return substitutions


def _canonical_answer(value: Any) -> str:
    result = sp.simplify(value)
    if result.free_symbols:
        raise ToolExecutionError(f"plugin result still has symbols: {result}")
    if result.is_number is not True or result.is_real is False:
        raise ToolExecutionError(f"plugin result is not a real number: {result}")
    return sp.sstr(result)


def _equations(value: Any) -> list[sp.Expr]:
    if not isinstance(value, list) or len(value) < 1:
        raise ToolArgumentError("equations must be a non-empty list")
    return [_expression(item, "equations") for item in value]


def solve_by_elimination(args: Any) -> str:
    values = _require_keys(args, {"equations", "target"})
    equations = _equations(values["equations"])
    target_name = values["target"]
    if not isinstance(target_name, str) or not target_name.isidentifier():
        raise ToolArgumentError("target must be a variable name")
    symbols = sorted(
        set().union(*(equation.free_symbols for equation in equations)),
        key=str,
    )
    target = sp.Symbol(target_name)
    if target not in symbols:
        raise ToolArgumentError("target does not occur in equations")
    solutions = sp.solve(equations, symbols, dict=True)
    if len(solutions) != 1 or target not in solutions[0]:
        raise ToolExecutionError("system does not have one target solution")
    return _canonical_answer(solutions[0][target])


def solve_by_substitution(args: Any) -> str:
    values = _require_mapping(args)
    mode = values.get("mode")
    if mode == "solve_system":
        values = _require_keys(values, {"mode", "equations", "target"})
        return solve_by_elimination(
            {"equations": values["equations"], "target": values["target"]}
        )
    if mode == "evaluate":
        values = _require_keys(values, {"mode", "expression", "substitutions"})
        expression = _expression(values["expression"], "expression")
        result = expression.subs(_substitutions(values["substitutions"]))
        return _canonical_answer(result)
    raise ToolArgumentError("substitution mode must be solve_system or evaluate")


def solve_by_factorization(args: Any) -> str:
    values = _require_mapping(args)
    mode = values.get("mode")
    if mode == "evaluate":
        values = _require_keys(values, {"mode", "expression", "substitutions"})
        expression = sp.factor(_expression(values["expression"], "expression"))
        return _canonical_answer(
            expression.subs(_substitutions(values["substitutions"]))
        )
    if mode == "root_power_sum":
        values = _require_keys(
            values, {"mode", "polynomial", "variable", "power"}
        )
        variable_name = values["variable"]
        if not isinstance(variable_name, str) or not variable_name.isidentifier():
            raise ToolArgumentError("variable must be a variable name")
        power = values["power"]
        if not isinstance(power, int) or isinstance(power, bool) or power <= 0:
            raise ToolArgumentError("power must be a positive integer")
        variable = sp.Symbol(variable_name)
        polynomial = sp.factor(_expression(values["polynomial"], "polynomial"))
        roots = sp.solve(polynomial, variable)
        if not roots:
            raise ToolExecutionError("polynomial has no roots")
        return _canonical_answer(sum(root**power for root in roots))
    raise ToolArgumentError("factorization mode must be evaluate or root_power_sum")


def solve_by_expansion(args: Any) -> str:
    values = _require_mapping(args)
    mode = values.get("mode")
    if mode == "evaluate":
        values = _require_keys(values, {"mode", "expression", "substitutions"})
        expression = sp.expand(_expression(values["expression"], "expression"))
        return _canonical_answer(
            expression.subs(_substitutions(values["substitutions"]))
        )
    if mode == "symmetric_square_sum":
        values = _require_keys(values, {"mode", "sum", "product"})
        total = _number(values["sum"], "sum")
        product = _number(values["product"], "product")
        return _canonical_answer(sp.expand(total**2 - 2 * product))
    raise ToolArgumentError(
        "expansion mode must be evaluate or symmetric_square_sum"
    )


def solve_by_proportional_scaling(args: Any) -> str:
    values = _require_keys(
        args, {"base_quantity", "base_value", "target_quantity"}
    )
    base_quantity = _number(values["base_quantity"], "base_quantity")
    if base_quantity == 0:
        raise ToolArgumentError("base_quantity cannot be zero")
    result = (
        _number(values["base_value"], "base_value")
        * _number(values["target_quantity"], "target_quantity")
        / base_quantity
    )
    return _canonical_answer(result)


def solve_by_cross_multiplication(args: Any) -> str:
    values = _require_keys(args, {"left_num", "left_den", "right_num"})
    left_num = _number(values["left_num"], "left_num")
    if left_num == 0:
        raise ToolArgumentError("left_num cannot be zero")
    result = (
        _number(values["right_num"], "right_num")
        * _number(values["left_den"], "left_den")
        / left_num
    )
    return _canonical_answer(result)


def solve_by_iteration(args: Any) -> str:
    values = _require_keys(args, {"start", "step", "updates"})
    updates = values["updates"]
    if not isinstance(updates, int) or isinstance(updates, bool) or updates < 0:
        raise ToolArgumentError("updates must be a non-negative integer")
    value = _number(values["start"], "start")
    step = _number(values["step"], "step")
    for _ in range(updates):
        value = sp.simplify(value + step)
    return _canonical_answer(value)


TOOL_REGISTRY: dict[str, Callable[[Any], str]] = {
    "substitution": solve_by_substitution,
    "elimination": solve_by_elimination,
    "factorization": solve_by_factorization,
    "expansion": solve_by_expansion,
    "proportional_scaling": solve_by_proportional_scaling,
    "cross_multiplication": solve_by_cross_multiplication,
    "iteration": solve_by_iteration,
}


def execute_tool(tool: str, tool_args: Any) -> str:
    """Dispatch one deterministic tool or fail explicitly."""

    plugin = TOOL_REGISTRY.get(tool)
    if plugin is None:
        raise ToolArgumentError(f"unknown tool: {tool!r}")
    return plugin(tool_args)


def validate_tool_args(tool: str, tool_args: Any) -> None:
    """Validate schema and symbolic syntax without requiring execution success."""

    try:
        execute_tool(tool, tool_args)
    except ToolExecutionError:
        # Schema and symbolic parsing were valid; execution is a separate metric.
        return


def _expressions_equivalent(left: Any, right: Any) -> bool:
    try:
        left_expr, right_expr = sp.sympify(left), sp.sympify(right)
    except (sp.SympifyError, TypeError, ValueError):
        return False
    return bool(sp.simplify(left_expr - right_expr) == 0)


def _equations_equivalent(left: Any, right: Any) -> bool:
    """Treat zero-form equations as equivalent up to a nonzero scalar."""

    try:
        left_expr, right_expr = sp.sympify(left), sp.sympify(right)
    except (sp.SympifyError, TypeError, ValueError):
        return False
    if sp.simplify(left_expr - right_expr) == 0:
        return True
    if right_expr == 0:
        return False
    ratio = sp.simplify(left_expr / right_expr)
    return bool(not ratio.free_symbols and ratio != 0)


def _semantic_equal(left: Any, right: Any, field: str = "") -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return False
        return all(_semantic_equal(left[key], right[key], str(key)) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        if field == "equations":
            unmatched = list(right)
            for expression in left:
                for index, candidate in enumerate(unmatched):
                    if _equations_equivalent(expression, candidate):
                        unmatched.pop(index)
                        break
                else:
                    return False
            return not unmatched
        return all(
            _semantic_equal(left_item, right_item, field)
            for left_item, right_item in zip(left, right)
        )
    if field in {"expression", "polynomial"}:
        return _expressions_equivalent(left, right)
    if left == right:
        return True
    try:
        return bool(sp.simplify(sp.sympify(left) - sp.sympify(right)) == 0)
    except (sp.SympifyError, TypeError, ValueError):
        return False


def tool_args_semantically_equivalent(
    tool: str,
    predicted: Any,
    gold: Any,
) -> bool:
    """Compare tool arguments, accepting equivalent symbolic expressions."""

    try:
        validate_tool_args(tool, predicted)
        validate_tool_args(tool, gold)
    except PluginError:
        return False
    return _semantic_equal(predicted, gold)
