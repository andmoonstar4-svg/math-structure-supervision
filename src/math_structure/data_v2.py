"""Synthetic algebra data with a nonredundant structure/tool taxonomy.

``structure`` describes a mathematical relationship in the problem, while
``tool`` names one canonical operation used to exploit it. Each structure is
paired with multiple tools, and several tools apply to multiple structures.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, MutableSet, Sequence

import sympy as sp

from .tools import execute_tool


@dataclass(frozen=True)
class VariantSpec:
    key: str
    structure: str
    tool: str
    seen_templates: tuple[str, ...]
    unseen_templates: tuple[str, ...]


VARIANTS = (
    VariantSpec(
        key="linear_elimination",
        structure="two_linear_relations",
        tool="elimination",
        seen_templates=(
            "Solve the simultaneous relations {equation_1} and {equation_2}. Find {target}.",
            "Two linear constraints hold: {equation_1}; {equation_2}. Determine {target}.",
        ),
        unseen_templates=(
            "The lines {equation_1} and {equation_2} intersect. Give their {target}-coordinate.",
        ),
    ),
    VariantSpec(
        key="linear_substitution",
        structure="two_linear_relations",
        tool="substitution",
        seen_templates=(
            "Given {equation_1}, insert it into {equation_2} and find {target}.",
            "The linked relations are {equation_1} and {equation_2}. What is {target}?",
        ),
        unseen_templates=(
            "One variable is isolated in {equation_1}; use the other constraint {equation_2} to obtain {target}.",
        ),
    ),
    VariantSpec(
        key="identity_factorization",
        structure="polynomial_identity",
        tool="factorization",
        seen_templates=(
            "Use the product pattern for a difference of squares to compute {a}^2 - {b}^2.",
            "Evaluate {a}^2 - {b}^2 by rewriting it as a product.",
        ),
        unseen_templates=(
            "Without separately squaring both numbers, find the difference between {a} squared and {b} squared.",
        ),
    ),
    VariantSpec(
        key="identity_expansion",
        structure="polynomial_identity",
        tool="expansion",
        seen_templates=(
            "Expand the square and evaluate ({a} + {b})^2.",
            "Use the binomial-square identity to compute ({a} + {b})^2.",
        ),
        unseen_templates=(
            "What number results from opening the brackets in ({a} + {b})({a} + {b})?",
        ),
    ),
    VariantSpec(
        key="ratio_scaling",
        structure="equivalent_ratios",
        tool="proportional_scaling",
        seen_templates=(
            "At a fixed rate, {base_items} items cost {base_value} dollars. What do {target_items} items cost?",
            "Scale the ratio {base_items}:{base_value} to {target_items}:x. Find x.",
        ),
        unseen_templates=(
            "A batch of {base_items} units has value {base_value}. Find the value of a same-rate batch of {target_items} units.",
        ),
    ),
    VariantSpec(
        key="ratio_cross_multiplication",
        structure="equivalent_ratios",
        tool="cross_multiplication",
        seen_templates=(
            "Solve the proportion {left_num}/{left_den} = {right_num}/x for x.",
            "Cross-multiply {left_num}:{left_den} = {right_num}:x and determine x.",
        ),
        unseen_templates=(
            "Complete the equivalent fractions: {left_num}/{left_den} = {right_num}/[blank].",
        ),
    ),
    VariantSpec(
        key="recurrence_iteration",
        structure="additive_recurrence",
        tool="iteration",
        seen_templates=(
            "A sequence starts at {start} and changes by {step} each term. Iterate to find term {index}.",
            "Given a1 = {start} and a(n+1) = a(n) {step_text}, find a{index}.",
        ),
        unseen_templates=(
            "A machine begins at {start} and applies a change of {step} for {updates} rounds. What is its final value?",
        ),
    ),
    VariantSpec(
        key="recurrence_substitution",
        structure="additive_recurrence",
        tool="substitution",
        seen_templates=(
            "An arithmetic sequence has formula a(n) = {start} + (n - 1)({step}). Substitute n = {index}.",
            "Use the closed form b(n) = {start} + (n - 1)({step}) to find b({index}).",
        ),
        unseen_templates=(
            "The repeated-change process is summarized by c(k) = {start} + (k - 1)({step}). Evaluate it at k = {index}.",
        ),
    ),
    VariantSpec(
        key="symmetric_expansion",
        structure="symmetric_sum_product",
        tool="expansion",
        seen_templates=(
            "If x + y = {sum} and xy = {product}, use a square expansion to find x^2 + y^2.",
            "Two numbers have sum {sum} and product {product}. Compute the sum of their squares from the symmetric identity.",
        ),
        unseen_templates=(
            "Knowing only that p + q = {sum} and pq = {product}, determine p^2 + q^2 by opening (p + q)^2.",
        ),
    ),
    VariantSpec(
        key="symmetric_factorization",
        structure="symmetric_sum_product",
        tool="factorization",
        seen_templates=(
            "Factor {polynomial} = 0 to obtain its two roots, then give the sum of their squares.",
            "The roots of {polynomial} = 0 are r and s. Find r^2 + s^2 by factoring the quadratic.",
        ),
        unseen_templates=(
            "Split the quadratic {polynomial} into linear factors and report the squared-root sum.",
        ),
    ),
    VariantSpec(
        key="expression_substitution",
        structure="transformed_expression",
        tool="substitution",
        seen_templates=(
            "For f(t) = {polynomial}, substitute t = {value} and find f({value}).",
            "Evaluate the transformed expression {polynomial_z} at z = {value}.",
        ),
        unseen_templates=(
            "Insert u = {value} into {polynomial_u}. What value results?",
        ),
    ),
    VariantSpec(
        key="expression_expansion",
        structure="transformed_expression",
        tool="expansion",
        seen_templates=(
            "Expand ({variable} + {offset_1})({variable} + {offset_2}) and evaluate it at {variable} = {value}.",
            "Open the product ({variable} + {offset_1})({variable} + {offset_2}), then set {variable} = {value}.",
        ),
        unseen_templates=(
            "After multiplying out the two shifted factors ({variable} + {offset_1}) and ({variable} + {offset_2}), find the result when {variable} is {value}.",
        ),
    ),
)

_VARIANT_BY_KEY = {variant.key: variant for variant in VARIANTS}


def _plain_expression(expr: sp.Expr) -> str:
    return sp.sstr(sp.expand(expr)).replace("**", "^").replace("*", "")


def _sample_parameters(key: str, rng: random.Random) -> dict[str, Any]:
    if key == "linear_elimination":
        x_value, y_value = rng.randint(-9, 9), rng.randint(-9, 9)
        while True:
            a, b, c, d = (rng.randint(-5, 5) for _ in range(4))
            if a and b and c and d and a * d - b * c != 0:
                break
        return {
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "rhs_1": a * x_value + b * y_value,
            "rhs_2": c * x_value + d * y_value,
            "target": rng.choice(("x", "y")),
        }

    if key == "linear_substitution":
        x_value = rng.randint(-9, 9)
        m = rng.choice([value for value in range(-5, 6) if value])
        n = rng.randint(-8, 8)
        y_value = m * x_value + n
        while True:
            a = rng.choice([value for value in range(-5, 6) if value])
            b = rng.choice([value for value in range(-5, 6) if value])
            if a + b * m != 0:
                break
        return {
            "m": m,
            "n": n,
            "a": a,
            "b": b,
            "rhs": a * x_value + b * y_value,
            "target": rng.choice(("x", "y")),
        }

    if key == "identity_factorization":
        b = rng.randint(2, 45)
        return {"a": rng.randint(b + 1, 80), "b": b}

    if key == "identity_expansion":
        return {"a": rng.randint(2, 40), "b": rng.randint(2, 40)}

    if key == "ratio_scaling":
        factor = rng.randint(2, 12)
        base_items = rng.randint(2, 15)
        base_value = rng.randint(2, 25)
        return {
            "base_items": base_items,
            "base_value": base_value,
            "target_items": base_items * factor,
            "factor": factor,
        }

    if key == "ratio_cross_multiplication":
        factor = rng.randint(2, 12)
        left_num = rng.randint(2, 15)
        left_den = rng.randint(2, 20)
        return {
            "left_num": left_num,
            "left_den": left_den,
            "right_num": left_num * factor,
        }

    if key in {"recurrence_iteration", "recurrence_substitution"}:
        return {
            "start": rng.randint(-20, 20),
            "step": rng.choice([value for value in range(-9, 10) if value]),
            "index": rng.randint(3, 15),
        }

    if key in {"symmetric_expansion", "symmetric_factorization"}:
        root_1, root_2 = rng.randint(-15, 15), rng.randint(-15, 15)
        while root_1 == root_2:
            root_2 = rng.randint(-15, 15)
        return {
            "root_1": root_1,
            "root_2": root_2,
            "sum": root_1 + root_2,
            "product": root_1 * root_2,
        }

    if key == "expression_substitution":
        return {
            "a": rng.choice([value for value in range(-5, 6) if value]),
            "b": rng.randint(-8, 8),
            "c": rng.randint(-12, 12),
            "value": rng.randint(-7, 7),
        }

    if key == "expression_expansion":
        return {
            "offset_1": rng.randint(-10, 10),
            "offset_2": rng.randint(-10, 10),
            "value": rng.randint(-7, 7),
            "variable": rng.choice(("x", "t", "z")),
        }

    raise KeyError(f"Unknown variant: {key}")


def _candidate_answer(key: str, p: dict[str, Any]) -> int:
    if key == "linear_elimination":
        determinant = p["a"] * p["d"] - p["b"] * p["c"]
        x_value = (p["rhs_1"] * p["d"] - p["b"] * p["rhs_2"]) // determinant
        y_value = (p["a"] * p["rhs_2"] - p["rhs_1"] * p["c"]) // determinant
        return x_value if p["target"] == "x" else y_value
    if key == "linear_substitution":
        x_value = (p["rhs"] - p["b"] * p["n"]) // (p["a"] + p["b"] * p["m"])
        y_value = p["m"] * x_value + p["n"]
        return x_value if p["target"] == "x" else y_value
    if key == "identity_factorization":
        return p["a"] ** 2 - p["b"] ** 2
    if key == "identity_expansion":
        return (p["a"] + p["b"]) ** 2
    if key == "ratio_scaling":
        return p["base_value"] * p["factor"]
    if key == "ratio_cross_multiplication":
        return p["right_num"] * p["left_den"] // p["left_num"]
    if key in {"recurrence_iteration", "recurrence_substitution"}:
        return p["start"] + (p["index"] - 1) * p["step"]
    if key in {"symmetric_expansion", "symmetric_factorization"}:
        return p["root_1"] ** 2 + p["root_2"] ** 2
    if key == "expression_substitution":
        return p["a"] * p["value"] ** 2 + p["b"] * p["value"] + p["c"]
    if key == "expression_expansion":
        return (p["value"] + p["offset_1"]) * (p["value"] + p["offset_2"])
    raise KeyError(f"Unknown variant: {key}")


def _template_context(key: str, p: dict[str, Any]) -> dict[str, Any]:
    context = dict(p)
    x, y = sp.symbols("x y")
    if key == "linear_elimination":
        context["equation_1"] = f"{_plain_expression(p['a'] * x + p['b'] * y)} = {p['rhs_1']}"
        context["equation_2"] = f"{_plain_expression(p['c'] * x + p['d'] * y)} = {p['rhs_2']}"
    elif key == "linear_substitution":
        context["equation_1"] = f"y = {_plain_expression(p['m'] * x + p['n'])}"
        context["equation_2"] = f"{_plain_expression(p['a'] * x + p['b'] * y)} = {p['rhs']}"
    elif key in {"recurrence_iteration", "recurrence_substitution"}:
        context["updates"] = p["index"] - 1
        context["step_text"] = f"+ {p['step']}" if p["step"] > 0 else f"- {abs(p['step'])}"
    elif key == "symmetric_factorization":
        t = sp.Symbol("t")
        polynomial = t**2 - p["sum"] * t + p["product"]
        context["polynomial"] = _plain_expression(polynomial)
    elif key == "expression_substitution":
        for variable, field in (("t", "polynomial"), ("z", "polynomial_z"), ("u", "polynomial_u")):
            symbol = sp.Symbol(variable)
            expression = p["a"] * symbol**2 + p["b"] * symbol + p["c"]
            context[field] = _plain_expression(expression)
    return context


def verify_example(example: dict[str, Any]) -> bool:
    """Independently verify a v2 example with SymPy."""

    key = example["variant"]
    p = example["parameters"]
    answer = sp.sympify(example["answer"])

    if key == "linear_elimination":
        x, y = sp.symbols("x y")
        solutions = sp.solve(
            (
                sp.Eq(p["a"] * x + p["b"] * y, p["rhs_1"]),
                sp.Eq(p["c"] * x + p["d"] * y, p["rhs_2"]),
            ),
            (x, y),
            dict=True,
        )
        expected = solutions[0][x if p["target"] == "x" else y]
    elif key == "linear_substitution":
        x, y = sp.symbols("x y")
        solutions = sp.solve(
            (
                sp.Eq(y, p["m"] * x + p["n"]),
                sp.Eq(p["a"] * x + p["b"] * y, p["rhs"]),
            ),
            (x, y),
            dict=True,
        )
        if len(solutions) != 1:
            return False
        expected = solutions[0][x if p["target"] == "x" else y]
    elif key == "identity_factorization":
        u, v = sp.symbols("u v")
        expected = sp.factor(u**2 - v**2).subs({u: p["a"], v: p["b"]})
    elif key == "identity_expansion":
        expected = sp.expand((sp.Integer(p["a"]) + p["b"]) ** 2)
    elif key == "ratio_scaling":
        rate = sp.Rational(p["base_value"], p["base_items"])
        expected = rate * p["target_items"]
    elif key == "ratio_cross_multiplication":
        unknown = sp.Symbol("unknown")
        solutions = sp.solve(
            sp.Eq(
                sp.Rational(p["left_num"], p["left_den"]),
                sp.Rational(p["right_num"], 1) / unknown,
            ),
            unknown,
        )
        expected = solutions[0]
    elif key in {"recurrence_iteration", "recurrence_substitution"}:
        k = sp.Symbol("k", integer=True)
        expected = sp.Integer(p["start"]) + sp.summation(
            p["step"], (k, 1, p["index"] - 1)
        )
    elif key == "symmetric_expansion":
        expected = sp.Integer(p["sum"]) ** 2 - 2 * sp.Integer(p["product"])
    elif key == "symmetric_factorization":
        t = sp.Symbol("t")
        roots = sp.solve(t**2 - p["sum"] * t + p["product"], t)
        expected = sum(root**2 for root in roots)
    elif key == "expression_substitution":
        t = sp.Symbol("t")
        expected = (p["a"] * t**2 + p["b"] * t + p["c"]).subs(t, p["value"])
    elif key == "expression_expansion":
        symbol = sp.Symbol(p["variable"])
        expression = sp.expand(
            (symbol + p["offset_1"]) * (symbol + p["offset_2"])
        )
        expected = expression.subs(symbol, p["value"])
    else:
        return False
    return bool(sp.simplify(expected - answer) == 0)


def analyze_taxonomy(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    structure_to_tools: dict[str, set[str]] = {}
    tool_to_structures: dict[str, set[str]] = {}
    for row in rows:
        structure_to_tools.setdefault(row["structure"], set()).add(row["tool"])
        tool_to_structures.setdefault(row["tool"], set()).add(row["structure"])

    structure_has_choices = any(len(tools) > 1 for tools in structure_to_tools.values())
    tool_is_reused = any(
        len(structures) > 1 for structures in tool_to_structures.values()
    )
    is_many_to_many = structure_has_choices and tool_is_reused
    warning = None
    if not is_many_to_many:
        warning = (
            "Taxonomy v2 does not satisfy the intended many-to-many mapping: "
            "at least one structure must have multiple tools and at least one tool "
            "must apply to multiple structures."
        )
    return {
        "structure_to_tools": {
            key: sorted(values) for key, values in sorted(structure_to_tools.items())
        },
        "tool_to_structures": {
            key: sorted(values) for key, values in sorted(tool_to_structures.items())
        },
        "has_structure_with_multiple_tools": structure_has_choices,
        "has_tool_with_multiple_structures": tool_is_reused,
        "is_many_to_many": is_many_to_many,
        "is_one_to_one": bool(structure_to_tools)
        and all(len(values) == 1 for values in structure_to_tools.values())
        and all(len(values) == 1 for values in tool_to_structures.values()),
        "warning": warning,
    }


def _build_tool_args(key: str, p: dict[str, Any]) -> dict[str, Any]:
    """Construct deterministic plugin inputs directly from generator state."""

    x, y = sp.symbols("x y")
    if key == "linear_elimination":
        return {
            "equations": [
                sp.sstr(p["a"] * x + p["b"] * y - p["rhs_1"]),
                sp.sstr(p["c"] * x + p["d"] * y - p["rhs_2"]),
            ],
            "target": p["target"],
        }
    if key == "linear_substitution":
        return {
            "mode": "solve_system",
            "equations": [
                sp.sstr(y - (p["m"] * x + p["n"])),
                sp.sstr(p["a"] * x + p["b"] * y - p["rhs"]),
            ],
            "target": p["target"],
        }
    if key == "identity_factorization":
        return {
            "mode": "evaluate",
            "expression": "u**2 - v**2",
            "substitutions": {"u": p["a"], "v": p["b"]},
        }
    if key == "identity_expansion":
        return {
            "mode": "evaluate",
            "expression": "(a + b)**2",
            "substitutions": {"a": p["a"], "b": p["b"]},
        }
    if key == "ratio_scaling":
        return {
            "base_quantity": p["base_items"],
            "base_value": p["base_value"],
            "target_quantity": p["target_items"],
        }
    if key == "ratio_cross_multiplication":
        return {
            "left_num": p["left_num"],
            "left_den": p["left_den"],
            "right_num": p["right_num"],
        }
    if key == "recurrence_iteration":
        return {
            "start": p["start"],
            "step": p["step"],
            "updates": p["index"] - 1,
        }
    if key == "recurrence_substitution":
        return {
            "mode": "evaluate",
            "expression": f"{p['start']} + (n - 1)*({p['step']})",
            "substitutions": {"n": p["index"]},
        }
    if key == "symmetric_expansion":
        return {
            "mode": "symmetric_square_sum",
            "sum": p["sum"],
            "product": p["product"],
        }
    if key == "symmetric_factorization":
        t = sp.Symbol("t")
        return {
            "mode": "root_power_sum",
            "polynomial": sp.sstr(t**2 - p["sum"] * t + p["product"]),
            "variable": "t",
            "power": 2,
        }
    if key == "expression_substitution":
        t = sp.Symbol("t")
        return {
            "mode": "evaluate",
            "expression": sp.sstr(p["a"] * t**2 + p["b"] * t + p["c"]),
            "substitutions": {"t": p["value"]},
        }
    if key == "expression_expansion":
        variable = sp.Symbol(p["variable"])
        return {
            "mode": "evaluate",
            "expression": sp.sstr(
                (variable + p["offset_1"]) * (variable + p["offset_2"])
            ),
            "substitutions": {p["variable"]: p["value"]},
        }
    raise KeyError(f"Unknown variant: {key}")


def _make_example(
    variant: VariantSpec,
    parameters: dict[str, Any],
    template_split: str,
    template_index: int,
    dataset_split: str,
    example_index: int,
) -> dict[str, Any]:
    templates = (
        variant.seen_templates
        if template_split == "seen"
        else variant.unseen_templates
    )
    answer = str(_candidate_answer(variant.key, parameters))
    tool_args = _build_tool_args(variant.key, parameters)
    compact_args = json.dumps(
        tool_args, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    example = {
        "id": f"{dataset_split}_{example_index:05d}",
        "split": dataset_split,
        "template_split": template_split,
        "template_id": f"{variant.key}.{template_split}_{template_index}",
        "taxonomy_version": "v2",
        "variant": variant.key,
        "structure": variant.structure,
        "tool": variant.tool,
        "problem": templates[template_index].format_map(
            _template_context(variant.key, parameters)
        ),
        "answer": answer,
        "structure_target": variant.structure,
        "tool_target": variant.tool,
        "baseline_target": answer,
        "oracle_target": answer,
        "structured_target": (
            f"STRUCTURE: {variant.structure}\n"
            f"TOOL: {variant.tool}\n"
            f"ANSWER: {answer}"
        ),
        "tool_args": tool_args,
        "routing_target": (
            f"STRUCTURE: {variant.structure}\n"
            f"TOOL: {variant.tool}\n"
            f"ARGUMENTS: {compact_args}"
        ),
        "parameters": parameters,
    }
    if not verify_example(example):
        raise ValueError(f"SymPy verification failed for {example['id']}: {example}")
    plugin_answer = execute_tool(variant.tool, tool_args)
    if plugin_answer != answer:
        raise ValueError(
            f"Plugin verification failed for {example['id']}: "
            f"{plugin_answer!r} != {answer!r}"
        )
    example["verification"] = {
        "engine": "sympy",
        "passed": True,
        "plugin_passed": True,
    }
    return example


def generate_examples(
    count: int,
    template_split: str,
    rng: random.Random,
    used_signatures: MutableSet[str] | None = None,
    dataset_split: str | None = None,
) -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if template_split not in {"seen", "unseen"}:
        raise ValueError("template_split must be 'seen' or 'unseen'")

    used = used_signatures if used_signatures is not None else set()
    split_name = dataset_split or (
        "train" if template_split == "seen" else "test_unseen"
    )
    rows: list[dict[str, Any]] = []
    for index in range(count):
        variant = VARIANTS[index % len(VARIANTS)]
        cycle = index // len(VARIANTS)
        templates = (
            variant.seen_templates
            if template_split == "seen"
            else variant.unseen_templates
        )
        template_index = cycle % len(templates)

        for _ in range(10_000):
            parameters = _sample_parameters(variant.key, rng)
            signature = json.dumps(
                [variant.key, parameters], sort_keys=True, separators=(",", ":")
            )
            if signature not in used:
                used.add(signature)
                break
        else:
            raise RuntimeError(f"Could not sample unique parameters for {variant.key}")

        rows.append(
            _make_example(
                variant,
                parameters,
                template_split,
                template_index,
                split_name,
                index,
            )
        )
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def generate_dataset_v2(
    output_dir: str | Path,
    seed: int = 42,
    train_sizes: Sequence[int] = (50, 100, 200, 500),
    seen_test_size: int = 120,
    unseen_test_size: int = 120,
) -> dict[str, Path]:
    """Generate nested v2 splits and enforce the many-to-many taxonomy."""

    sizes = tuple(sorted(set(int(size) for size in train_sizes)))
    if not sizes or sizes[0] <= 0:
        raise ValueError("train_sizes must contain positive integers")
    if seen_test_size <= 0 or unseen_test_size <= 0:
        raise ValueError("test sizes must be positive")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    used_signatures: set[str] = set()
    train_rows = generate_examples(
        max(sizes), "seen", random.Random(seed), used_signatures, "train"
    )
    seen_rows = generate_examples(
        seen_test_size,
        "seen",
        random.Random(seed + 1),
        used_signatures,
        "test_seen",
    )
    unseen_rows = generate_examples(
        unseen_test_size,
        "unseen",
        random.Random(seed + 2),
        used_signatures,
        "test_unseen",
    )

    mapping = analyze_taxonomy(train_rows + seen_rows + unseen_rows)
    if not mapping["is_many_to_many"]:
        raise ValueError(mapping["warning"])

    paths: dict[str, Path] = {}
    for size in sizes:
        path = destination / f"train_{size}.jsonl"
        _write_jsonl(path, train_rows[:size])
        paths[f"train_{size}"] = path
    paths["test_seen"] = destination / "test_seen.jsonl"
    paths["test_unseen"] = destination / "test_unseen.jsonl"
    _write_jsonl(paths["test_seen"], seen_rows)
    _write_jsonl(paths["test_unseen"], unseen_rows)

    paths["label_mapping"] = destination / "label_mapping.json"
    paths["label_mapping"].write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metadata = {
        "taxonomy_version": "v2",
        "taxonomy_definitions": {
            "structure": "a mathematical relationship or pattern present in the problem",
            "tool": "a canonical operation or solving strategy applied to exploit that structure",
        },
        "seed": seed,
        "train_sizes": list(sizes),
        "seen_test_size": seen_test_size,
        "unseen_test_size": unseen_test_size,
        "train_sets_are_nested": True,
        "variants": [
            {
                "key": variant.key,
                "structure": variant.structure,
                "tool": variant.tool,
                "seen_template_ids": [
                    f"{variant.key}.seen_{index}"
                    for index in range(len(variant.seen_templates))
                ],
                "unseen_template_ids": [
                    f"{variant.key}.unseen_{index}"
                    for index in range(len(variant.unseen_templates))
                ],
            }
            for variant in VARIANTS
        ],
        "target_formats": {
            "baseline": "<exact number>",
            "structure_aware": "STRUCTURE: <label>\\nTOOL: <label>\\nANSWER: <exact number>",
            "oracle": "<exact number>; gold structure and tool are included in the prompt",
            "tool_routing": "STRUCTURE: <label>\\nTOOL: <label>\\nARGUMENTS: <JSON object>",
        },
        "verification": (
            "Every generated answer passed an independent SymPy check and "
            "matched deterministic plugin(tool, tool_args) execution."
        ),
    }
    paths["metadata"] = destination / "metadata.json"
    paths["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths
