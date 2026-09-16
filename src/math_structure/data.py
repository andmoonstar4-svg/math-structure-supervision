"""Synthetic, SymPy-verified algebra data used by the experiment.

The generator deliberately keeps every answer to a single exact number.  That
makes answer extraction and exact-match evaluation unambiguous, while the
surface templates still let us test generalization to unseen wording.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, MutableSet, Sequence

import sympy as sp


@dataclass(frozen=True)
class StructureSpec:
    """Names, tool label, and surface forms for one algebraic structure."""

    name: str
    tool: str
    seen_templates: tuple[str, ...]
    unseen_templates: tuple[str, ...]


STRUCTURE_SPECS: tuple[StructureSpec, ...] = (
    StructureSpec(
        name="symmetric_polynomial_identity",
        tool="expand_symmetric_identity",
        seen_templates=(
            "If x + y = {sum} and xy = {product}, find x^2 + y^2.",
            "Two numbers have sum {sum} and product {product}. What is the sum of their squares?",
            "Given s = x + y = {sum} and p = xy = {product}, compute x^2 + y^2.",
        ),
        unseen_templates=(
            "Without solving for x and y, evaluate x^2 + y^2 when x + y = {sum} and xy = {product}.",
            "The roots x and y of t^2 - ({sum})t + ({product}) = 0 are real. Find x^2 + y^2.",
        ),
    ),
    StructureSpec(
        name="factorization",
        tool="factor_difference_of_squares",
        seen_templates=(
            "Evaluate {a}^2 - {b}^2.",
            "Compute the difference of squares: {a} squared minus {b} squared.",
            "What is {a}^2 - {b}^2? Use factorization if helpful.",
        ),
        unseen_templates=(
            "Using (u - v)(u + v), calculate {a}^2 - {b}^2.",
            "A square of side {b} is removed from a square of side {a}. What area remains?",
        ),
    ),
    StructureSpec(
        name="substitution",
        tool="substitute_expression",
        seen_templates=(
            "Let f(t) = {polynomial}. Find f({value}).",
            "Substitute z = {value} into {polynomial_z}.",
            "For P(x) = {polynomial_x}, evaluate P({value}).",
        ),
        unseen_templates=(
            "What number does {polynomial_u} produce when u is {value}?",
            "A rule maps q to {polynomial_q}. Determine its output at q = {value}.",
        ),
    ),
    StructureSpec(
        name="linear_elimination",
        tool="solve_linear_system",
        seen_templates=(
            "Solve {equation_1} and {equation_2}. What is {target}?",
            "The equations {equation_1} and {equation_2} hold simultaneously. Find {target}.",
            "Use elimination on the system {equation_1}; {equation_2}. Determine {target}.",
        ),
        unseen_templates=(
            "The lines {equation_1} and {equation_2} intersect. Give the {target}-coordinate.",
            "Find the value of {target} satisfying both constraints: {equation_1} and {equation_2}.",
        ),
    ),
    StructureSpec(
        name="ratio_proportion",
        tool="solve_proportion",
        seen_templates=(
            "Solve the proportion {left_num}/{left_den} = {right_num}/x for x.",
            "If {left_num}:{left_den} = {right_num}:x, what is x?",
            "The ratio {left_num} to {left_den} equals the ratio {right_num} to x. Find x.",
        ),
        unseen_templates=(
            "At a fixed rate, {left_num} items cost {left_den} dollars. What do {right_num} items cost?",
            "Complete the equivalent fractions: {left_num}/{left_den} = {right_num}/[blank].",
        ),
    ),
    StructureSpec(
        name="recurrence",
        tool="unroll_recurrence",
        seen_templates=(
            "A sequence starts with a1 = {start} and each next term is the current term {step_text}. Find a{index}.",
            "Starting at {start}, repeatedly change the value by {step}. What is term {index}, counting the start as term 1?",
            "An arithmetic sequence has first term {start} and common difference {step}. Find its {index}th term.",
        ),
        unseen_templates=(
            "A machine begins at {start}; after each round it changes its value by {step}. What value appears after {updates} updates?",
            "Unroll b(n+1) = b(n) {step_text}, with b(1) = {start}, to obtain b({index}).",
        ),
    ),
)

_SPEC_BY_NAME = {spec.name: spec for spec in STRUCTURE_SPECS}


def _plain_expression(expr: sp.Expr) -> str:
    """Render a tiny polynomial in a compact, model-friendly form."""

    return sp.sstr(sp.expand(expr)).replace("**", "^").replace("*", "")


def _sample_parameters(structure: str, rng: random.Random) -> dict[str, Any]:
    if structure == "symmetric_polynomial_identity":
        # Sampling actual roots guarantees that the stated real roots exist.
        x_value = rng.randint(-15, 15)
        y_value = rng.randint(-15, 15)
        while x_value == y_value:
            y_value = rng.randint(-15, 15)
        return {"sum": x_value + y_value, "product": x_value * y_value}

    if structure == "factorization":
        b = rng.randint(2, 45)
        a = rng.randint(b + 1, 80)
        return {"a": a, "b": b}

    if structure == "substitution":
        a = rng.choice([value for value in range(-5, 6) if value])
        b = rng.randint(-8, 8)
        c = rng.randint(-12, 12)
        value = rng.randint(-7, 7)
        return {"a": a, "b": b, "c": c, "value": value}

    if structure == "linear_elimination":
        x_value = rng.randint(-9, 9)
        y_value = rng.randint(-9, 9)
        while True:
            a, b, c, d = (rng.randint(-5, 5) for _ in range(4))
            if a and b and c and d and a * d - b * c != 0:
                break
        target = rng.choice(("x", "y"))
        return {
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "rhs_1": a * x_value + b * y_value,
            "rhs_2": c * x_value + d * y_value,
            "target": target,
        }

    if structure == "ratio_proportion":
        left_num = rng.randint(2, 15)
        left_den = rng.randint(2, 20)
        multiplier = rng.randint(2, 12)
        return {
            "left_num": left_num,
            "left_den": left_den,
            "right_num": left_num * multiplier,
        }

    if structure == "recurrence":
        step = rng.choice([value for value in range(-9, 10) if value])
        return {
            "start": rng.randint(-20, 20),
            "step": step,
            "index": rng.randint(3, 15),
        }

    raise KeyError(f"Unknown structure: {structure}")


def _candidate_answer(structure: str, parameters: dict[str, Any]) -> int:
    """Compute the intended answer with elementary formulas, before verification."""

    p = parameters
    if structure == "symmetric_polynomial_identity":
        return p["sum"] ** 2 - 2 * p["product"]
    if structure == "factorization":
        return p["a"] ** 2 - p["b"] ** 2
    if structure == "substitution":
        return p["a"] * p["value"] ** 2 + p["b"] * p["value"] + p["c"]
    if structure == "linear_elimination":
        determinant = p["a"] * p["d"] - p["b"] * p["c"]
        x_value = (p["rhs_1"] * p["d"] - p["b"] * p["rhs_2"]) // determinant
        y_value = (p["a"] * p["rhs_2"] - p["rhs_1"] * p["c"]) // determinant
        return x_value if p["target"] == "x" else y_value
    if structure == "ratio_proportion":
        return p["right_num"] * p["left_den"] // p["left_num"]
    if structure == "recurrence":
        return p["start"] + (p["index"] - 1) * p["step"]
    raise KeyError(f"Unknown structure: {structure}")


def _template_context(structure: str, parameters: dict[str, Any]) -> dict[str, Any]:
    context = dict(parameters)

    if structure == "substitution":
        for variable, key in (("t", "polynomial"), ("z", "polynomial_z"), ("x", "polynomial_x"), ("u", "polynomial_u"), ("q", "polynomial_q")):
            symbol = sp.Symbol(variable)
            expression = parameters["a"] * symbol**2 + parameters["b"] * symbol + parameters["c"]
            context[key] = _plain_expression(expression)

    elif structure == "linear_elimination":
        x, y = sp.symbols("x y")
        lhs_1 = parameters["a"] * x + parameters["b"] * y
        lhs_2 = parameters["c"] * x + parameters["d"] * y
        context["equation_1"] = f"{_plain_expression(lhs_1)} = {parameters['rhs_1']}"
        context["equation_2"] = f"{_plain_expression(lhs_2)} = {parameters['rhs_2']}"

    elif structure == "recurrence":
        step = parameters["step"]
        context["step_text"] = f"+ {step}" if step > 0 else f"- {abs(step)}"
        context["updates"] = parameters["index"] - 1

    return context


def verify_example(example: dict[str, Any]) -> bool:
    """Independently reconstruct and check one answer using SymPy."""

    structure = example["structure"]
    p = example["parameters"]
    answer = sp.sympify(str(example["answer"]), evaluate=True)

    if structure == "symmetric_polynomial_identity":
        x, y = sp.symbols("x y")
        identity_holds = sp.expand((x + y) ** 2 - 2 * x * y - (x**2 + y**2)) == 0
        expected = sp.Integer(p["sum"]) ** 2 - 2 * sp.Integer(p["product"])
        return bool(identity_holds and sp.simplify(expected - answer) == 0)

    if structure == "factorization":
        u, v = sp.symbols("u v")
        factored = sp.factor(u**2 - v**2)
        expected = factored.subs({u: p["a"], v: p["b"]})
        return bool(sp.simplify(expected - answer) == 0)

    if structure == "substitution":
        t = sp.Symbol("t")
        expression = p["a"] * t**2 + p["b"] * t + p["c"]
        expected = expression.subs(t, p["value"])
        return bool(sp.simplify(expected - answer) == 0)

    if structure == "linear_elimination":
        x, y = sp.symbols("x y")
        equations = (
            sp.Eq(p["a"] * x + p["b"] * y, p["rhs_1"]),
            sp.Eq(p["c"] * x + p["d"] * y, p["rhs_2"]),
        )
        solutions = sp.solve(equations, (x, y), dict=True)
        if len(solutions) != 1:
            return False
        expected = solutions[0][x if p["target"] == "x" else y]
        return bool(sp.simplify(expected - answer) == 0)

    if structure == "ratio_proportion":
        x = sp.Symbol("x")
        equation = sp.Eq(sp.Rational(p["left_num"], p["left_den"]), sp.Rational(p["right_num"], 1) / x)
        solutions = sp.solve(equation, x)
        return bool(len(solutions) == 1 and sp.simplify(solutions[0] - answer) == 0)

    if structure == "recurrence":
        k = sp.Symbol("k", integer=True)
        expected = sp.Integer(p["start"]) + sp.summation(p["step"], (k, 1, p["index"] - 1))
        return bool(sp.simplify(expected - answer) == 0)

    return False


def _make_example(
    spec: StructureSpec,
    parameters: dict[str, Any],
    template_split: str,
    template_index: int,
    dataset_split: str,
    example_index: int,
) -> dict[str, Any]:
    templates = spec.seen_templates if template_split == "seen" else spec.unseen_templates
    problem = templates[template_index].format_map(_template_context(spec.name, parameters))
    answer = str(_candidate_answer(spec.name, parameters))
    structure_target = spec.name
    tool_target = spec.tool
    example = {
        "id": f"{dataset_split}_{example_index:05d}",
        "split": dataset_split,
        "template_split": template_split,
        "template_id": f"{spec.name}.{template_split}_{template_index}",
        "structure": structure_target,
        "tool": tool_target,
        "problem": problem,
        "answer": answer,
        "structure_target": structure_target,
        "tool_target": tool_target,
        "baseline_target": answer,
        "oracle_target": answer,
        "structured_target": f"STRUCTURE: {structure_target}\nTOOL: {tool_target}\nANSWER: {answer}",
        "parameters": parameters,
    }
    passed = verify_example(example)
    if not passed:
        raise ValueError(f"SymPy verification failed for {example['id']}: {example}")
    example["verification"] = {"engine": "sympy", "passed": True}
    return example


def generate_examples(
    count: int,
    template_split: str,
    rng: random.Random,
    used_signatures: MutableSet[str] | None = None,
    dataset_split: str | None = None,
) -> list[dict[str, Any]]:
    """Generate a balanced set from seen or held-out surface templates.

    ``used_signatures`` can be shared across calls to prevent a numerical
    parameterization from leaking between training and test sets.
    """

    if count < 0:
        raise ValueError("count must be non-negative")
    if template_split not in {"seen", "unseen"}:
        raise ValueError("template_split must be 'seen' or 'unseen'")

    used = used_signatures if used_signatures is not None else set()
    split_name = dataset_split or ("train" if template_split == "seen" else "test_unseen")
    examples: list[dict[str, Any]] = []

    for index in range(count):
        spec = STRUCTURE_SPECS[index % len(STRUCTURE_SPECS)]
        structure_cycle = index // len(STRUCTURE_SPECS)
        templates = spec.seen_templates if template_split == "seen" else spec.unseen_templates
        template_index = structure_cycle % len(templates)

        for _ in range(10_000):
            parameters = _sample_parameters(spec.name, rng)
            signature = json.dumps([spec.name, parameters], sort_keys=True, separators=(",", ":"))
            if signature not in used:
                used.add(signature)
                break
        else:
            raise RuntimeError(f"Could not find a unique parameterization for {spec.name}")

        examples.append(
            _make_example(
                spec=spec,
                parameters=parameters,
                template_split=template_split,
                template_index=template_index,
                dataset_split=split_name,
                example_index=index,
            )
        )

    return examples


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read one generated JSONL file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def generate_dataset(
    output_dir: str | Path,
    seed: int = 42,
    train_sizes: Sequence[int] = (50, 100, 200, 500),
    seen_test_size: int = 120,
    unseen_test_size: int = 120,
) -> dict[str, Path]:
    """Generate nested train subsets and seen/unseen-template tests."""

    sizes = tuple(sorted(set(int(size) for size in train_sizes)))
    if not sizes or sizes[0] <= 0:
        raise ValueError("train_sizes must contain positive integers")
    if seen_test_size <= 0 or unseen_test_size <= 0:
        raise ValueError("test sizes must be positive")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    used_signatures: set[str] = set()

    train_examples = generate_examples(
        max(sizes),
        template_split="seen",
        rng=random.Random(seed),
        used_signatures=used_signatures,
        dataset_split="train",
    )
    seen_examples = generate_examples(
        seen_test_size,
        template_split="seen",
        rng=random.Random(seed + 1),
        used_signatures=used_signatures,
        dataset_split="test_seen",
    )
    unseen_examples = generate_examples(
        unseen_test_size,
        template_split="unseen",
        rng=random.Random(seed + 2),
        used_signatures=used_signatures,
        dataset_split="test_unseen",
    )

    paths: dict[str, Path] = {}
    for size in sizes:
        path = destination / f"train_{size}.jsonl"
        _write_jsonl(path, train_examples[:size])
        paths[f"train_{size}"] = path

    paths["test_seen"] = destination / "test_seen.jsonl"
    paths["test_unseen"] = destination / "test_unseen.jsonl"
    _write_jsonl(paths["test_seen"], seen_examples)
    _write_jsonl(paths["test_unseen"], unseen_examples)

    metadata = {
        "seed": seed,
        "train_sizes": list(sizes),
        "seen_test_size": seen_test_size,
        "unseen_test_size": unseen_test_size,
        "train_sets_are_nested": True,
        "structures": [
            {
                "name": spec.name,
                "tool": spec.tool,
                "seen_template_ids": [f"{spec.name}.seen_{i}" for i in range(len(spec.seen_templates))],
                "unseen_template_ids": [f"{spec.name}.unseen_{i}" for i in range(len(spec.unseen_templates))],
            }
            for spec in STRUCTURE_SPECS
        ],
        "target_formats": {
            "baseline": "<exact number>",
            "structure_aware": "STRUCTURE: <label>\\nTOOL: <label>\\nANSWER: <exact number>",
            "oracle": "<exact number>; gold structure and tool are included in the prompt",
        },
        "verification": "Every generated answer passed an independent SymPy check.",
    }
    paths["metadata"] = destination / "metadata.json"
    with paths["metadata"].open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    return paths
