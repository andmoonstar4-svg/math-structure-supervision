# Structure supervision for small-data algebra reasoning

This repository is a minimal experiment for the hypothesis that explicitly
supervising mathematical structure improves sample efficiency when training
on little data.

The original experiment compares three direct-generation conditions built from
the same synthetic algebra problems:

- **Baseline:** `problem -> answer`
- **Structure-aware:** `problem -> structure -> tool -> answer`
- **Oracle structure/tool:** `problem + gold structure + gold tool -> answer`

The fourth-round extension adds a hybrid condition:

- **Tool routing:** `problem -> structure + tool + arguments -> deterministic plugin -> answer`

The experiment is deliberately small: one causal language model, LoRA
adapters, deterministic programmatic data generation, and CSV outputs. It is a
proof of concept, not a benchmark framework.

## What is generated

The generator covers six algebraic structures:

1. symmetric polynomial identity
2. factorization
3. substitution
4. linear elimination
5. ratio/proportion
6. recurrence

Each structure has several natural-language templates and randomized numeric
parameters. SymPy checks every generated answer before it is written. Training
sets contain 50, 100, 200, and 500 examples. They are nested, so the 50-example
set is a subset of the 100-example set, and so on.

Two test splits isolate different kinds of generalization:

- `test_seen.jsonl` uses templates that occur in training, with new parameters.
- `test_unseen.jsonl` uses surface templates held out from every training set.

The JSONL rows retain the template, structure, tool, and verification metadata
so that generation and scoring are auditable.

In the original direct-generation experiment, `tool` is a supervised
symbolic-method label such as
`solve_linear_system` or `factor_difference_of_squares`; it is predicted by
the model but not executed. The fourth round adds separately generated v2 rows
whose canonical tools are executable through the small plugin registry. SymPy
is used independently to reject any example whose stored answer is not correct.

## Setup

Python 3.10 or newer is recommended. A CUDA GPU makes the complete sweep much
faster, but the data and evaluation smoke tests run on CPU.

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

The first training run downloads
[`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct).
If PyTorch needs a platform-specific CUDA build, install it using the command
from the [PyTorch selector](https://pytorch.org/get-started/locally/) before
installing this project.

## Reproduce the experiment

Run these commands from the repository root.

### 1. Generate deterministic data

```bash
python scripts/generate_data.py \
  --output-dir data \
  --seed 42 \
  --seen-test-size 120 \
  --unseen-test-size 120
```

This creates:

```text
data/
  train_50.jsonl
  train_100.jsonl
  train_200.jsonl
  train_500.jsonl
  test_seen.jsonl
  test_unseen.jsonl
  metadata.json
```

In PowerShell, either enter the command on one line or replace each `\` line
continuation with a PowerShell backtick.

### 2. Train the baseline LoRA adapters

```bash
python scripts/train_baseline.py \
  --data-dir data \
  --output-dir outputs \
  --train-sizes 50 100 200 500
```

### 3. Train the structure-aware LoRA adapters

```bash
python scripts/train_structure.py \
  --data-dir data \
  --output-dir outputs \
  --train-sizes 50 100 200 500
```

The adapters and each run's configuration are stored under
`outputs/baseline/train_<N>/` and `outputs/structure_aware/train_<N>/`.

### 4. Train the oracle structure/tool LoRA adapters

```bash
python scripts/train_oracle.py \
  --data-dir data \
  --output-dir outputs \
  --train-sizes 50 100 200 500
```

The oracle prompt reads the `structure` and `tool` fields directly from each
dataset row. The model is supervised only on the final answer. Adapters are
stored under `outputs/oracle/train_<N>/`; all three conditions otherwise use
the same model, seed, epochs, batch size, learning rate, and LoRA defaults.

### 5. Evaluate every run

```bash
python scripts/evaluate.py \
  --data-dir data \
  --output-dir outputs \
  --results-csv results/metrics.csv \
  --predictions-csv results/predictions.csv \
  --comparison-csv results/comparison.csv \
  --conditional-csv results/conditional_metrics.csv \
  --label-mapping-json results/label_mapping.json \
  --train-sizes 50 100 200 500
```

The evaluator writes:

- `results/metrics.csv`: aggregate accuracy by setting, training-set size, and
  seen/unseen template split.
- `results/predictions.csv`: one row per model prediction for error analysis.
- `results/comparison.csv`: a wide table comparing baseline, structure-aware,
  and oracle answer accuracy alongside label and conditional execution metrics.
- `results/conditional_metrics.csv`: structure-aware answer accuracy conditioned
  on correct structure, correct tool, both correct, or either label being wrong.
- `results/label_mapping.json`: the observed structure-to-tool and
  tool-to-structure mappings over the complete generated dataset.

Reported metrics are exact answer accuracy, structure prediction accuracy, and
tool prediction accuracy. Structure/tool metrics are meaningful for the
structure-aware setting; baseline and oracle are trained to emit only an answer.
Answer extraction permits harmless surrounding whitespace (and optional outer
`$` delimiters), then requires an exact match to the canonical integer string.

To rerun the complete experiment with another causal model, pass the same
`--model-name` value to all three training scripts and evaluation:

```bash
MODEL=some-org/some-causal-model
python scripts/train_baseline.py --model-name "$MODEL" --train-sizes 50 100 200 500
python scripts/train_structure.py --model-name "$MODEL" --train-sizes 50 100 200 500
python scripts/train_oracle.py --model-name "$MODEL" --train-sizes 50 100 200 500
python scripts/evaluate.py --model-name "$MODEL" --train-sizes 50 100 200 500
```

The default remains `HuggingFaceTB/SmolLM2-135M-Instruct`. The replacement must
be compatible with `AutoModelForCausalLM` and LoRA. Common attention projection
names are detected automatically; for an unusual architecture, pass its module
leaf names with `--lora-target-modules` during training.

## Third-round experiments

The 135M runs learned the auxiliary labels but left every answer-generation
condition near 1–3% accuracy. Even gold labels did not help, so those runs were
dominated by a model-capacity or symbolic-execution floor. They also revealed
that every original structure had exactly one tool and vice versa.

The third round keeps two questions separate:

- **Experiment A — capacity:** `Qwen/Qwen2.5-0.5B-Instruct` on the unchanged
  original data and taxonomy.
- **Experiment B — representation:** the same Qwen model and training settings
  on the separate, nonredundant `data_v2/` dataset.

The v2 definition is deliberately semantic: `structure` is a mathematical
relationship or pattern, while `tool` is one canonical operation or strategy
used to exploit it. Its six structures and seven tools form 12 supervised
combinations. Every structure permits two tools, while substitution, expansion,
and factorization each apply to multiple structures. The generator refuses to
write a v2 dataset if this many-to-many property is lost.

### Experiment A: larger model, original taxonomy

```bash
MODEL=Qwen/Qwen2.5-0.5B-Instruct
OUT=outputs/qwen_original

python scripts/train_baseline.py --data-dir data --output-dir "$OUT" --model-name "$MODEL" --train-sizes 50 100 200 500 --bf16 --gradient-checkpointing
python scripts/train_structure.py --data-dir data --output-dir "$OUT" --model-name "$MODEL" --train-sizes 50 100 200 500 --bf16 --gradient-checkpointing
python scripts/train_oracle.py --data-dir data --output-dir "$OUT" --model-name "$MODEL" --train-sizes 50 100 200 500 --bf16 --gradient-checkpointing

python scripts/evaluate.py --data-dir data --output-dir "$OUT" --results-csv results_qwen_original/metrics.csv --predictions-csv results_qwen_original/predictions.csv --comparison-csv results_qwen_original/comparison.csv --conditional-csv results_qwen_original/conditional_metrics.csv --label-mapping-json results_qwen_original/label_mapping.json --structure-confusion-csv results_qwen_original/structure_confusion.csv --tool-confusion-csv results_qwen_original/tool_confusion.csv --train-sizes 50 100 200 500 --batch-size 32 --model-dtype bfloat16
```

### Experiment B: larger model, taxonomy v2

```bash
python scripts/generate_data_v2.py --output-dir data_v2 --seed 42 --train-sizes 50 100 200 500 --seen-test-size 120 --unseen-test-size 120

MODEL=Qwen/Qwen2.5-0.5B-Instruct
OUT=outputs/qwen_v2

python scripts/train_baseline.py --data-dir data_v2 --output-dir "$OUT" --model-name "$MODEL" --train-sizes 50 100 200 500 --bf16 --gradient-checkpointing
python scripts/train_structure.py --data-dir data_v2 --output-dir "$OUT" --model-name "$MODEL" --train-sizes 50 100 200 500 --bf16 --gradient-checkpointing
python scripts/train_oracle.py --data-dir data_v2 --output-dir "$OUT" --model-name "$MODEL" --train-sizes 50 100 200 500 --bf16 --gradient-checkpointing

python scripts/evaluate.py --data-dir data_v2 --output-dir "$OUT" --results-csv results_qwen_v2/metrics.csv --predictions-csv results_qwen_v2/predictions.csv --comparison-csv results_qwen_v2/comparison.csv --conditional-csv results_qwen_v2/conditional_metrics.csv --label-mapping-json results_qwen_v2/label_mapping.json --structure-confusion-csv results_qwen_v2/structure_confusion.csv --tool-confusion-csv results_qwen_v2/tool_confusion.csv --train-sizes 50 100 200 500 --batch-size 32 --model-dtype bfloat16
```

The two result directories must be interpreted independently because Experiment
A changes capacity while Experiment B also changes the data taxonomy and task
mix. The observed learning curves and cautious interpretation are recorded in
`results_round3_summary.md`.

## Fourth-round hybrid tool-routing experiment

The fourth round keeps the Qwen 0.5B model, taxonomy-v2 examples, seed, LoRA
configuration, and seen/unseen splits fixed, but separates recognition from
execution:

```text
LLM:                  problem -> structure + tool + structured arguments
deterministic solver: tool + structured arguments -> canonical answer
```

This follows directly from the earlier diagnostics: the small models learned
structure/tool labels more reliably than direct symbolic execution, and giving
gold labels to a direct answer model did not substantially lift answer
accuracy. The hybrid condition therefore tests whether a small LLM can be a
useful recognizer and router even when it is a weak internal algebra engine.
It is a hybrid system, not pure end-to-end neural mathematical reasoning.

`src/math_structure/tools.py` contains one explicit, deterministic SymPy-backed
plugin for every canonical v2 tool label. Generated examples now include
programmatic `tool_args`; generation aborts unless executing the gold tool and
arguments reproduces the stored answer. Plugins reject unknown, missing, extra,
or malformed arguments instead of guessing.

Earlier artifacts remain untouched. The enriched data, new adapters, and
results are written to `data_v2_tools/`, `outputs/qwen_v2_hybrid/`, and
`results_hybrid/` respectively.

### 1. Generate plugin-ready v2 data

```bash
python scripts/generate_data_v2.py --output-dir data_v2_tools --seed 42 --train-sizes 50 100 200 500 --seen-test-size 120 --unseen-test-size 120
```

### 2. Train only the routing adapters

The baseline and structure-aware direct predictions are reused from the
matching third-round Qwen v2 runs; they are validated against every current
test-row ID, problem, and answer before scoring.

```bash
python scripts/train_tool_routing.py --data-dir data_v2_tools --output-dir outputs/qwen_v2_hybrid --train-sizes 50 100 200 500 --model-name Qwen/Qwen2.5-0.5B-Instruct --epochs 5 --batch-size 8 --learning-rate 2e-4 --max-length 256 --seed 17 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --bf16 --gradient-checkpointing
```

### 3. Evaluate all four systems

```bash
python scripts/evaluate_hybrid.py --data-dir data_v2_tools --routing-output-dir outputs/qwen_v2_hybrid --direct-predictions-csv results_qwen_v2/predictions.csv --metrics-csv results_hybrid/metrics.csv --predictions-csv results_hybrid/predictions.csv --train-sizes 50 100 200 500 --batch-size 32 --max-new-tokens 96 --model-dtype bfloat16
```

`results_hybrid/metrics.csv` compares baseline direct generation,
structure-aware direct generation, gold/oracle plugin execution, and predicted
plugin execution by training size and template split. It also reports routing,
argument, execution, and conditional metrics. `predictions.csv` retains raw
model output, parsed arguments, plugin answers, and one primary failure category
per failed route: `wrong_structure`, `wrong_tool`, `invalid_arguments`,
`valid_but_wrong_arguments`, `plugin_execution_failure`, or
`plugin_answer_mismatch`.

The oracle plugin should be treated as a pipeline integrity check and upper
bound, not a learned result. The central comparison is predicted-plugin versus
direct answer accuracy, especially on unseen templates. A hybrid advantage is
claimed only if those measured results support it.

In the recorded run, predicted-plugin answer accuracy rose from 10.0% to 98.3%
on seen templates and from 3.3% to 36.7% on unseen templates as the training set
grew from 50 to 500. At 500 examples, structure-aware direct generation reached
16.7% seen and 10.8% unseen. Oracle plugin accuracy was 100% in every cell, and
`P(final correct | tool and semantically correct arguments)` was also 100%
whenever that condition had support. Thus this run does support the narrow
hybrid-router hypothesis, while the much lower unseen score shows that routing
and argument extraction still generalize imperfectly. Full values and error
counts are in `results_hybrid/metrics.csv`, `results_hybrid/predictions.csv`, and
`results_hybrid/summary.md`.

On the tested RTX 4060 Laptop GPU with 8 GB VRAM, Qwen 0.5B fit with BF16,
training batch size 8, and gradient checkpointing. Evaluation batch size 32
also fit. If another 8 GB system runs out of memory, reduce evaluation batch
size to 16 or 8; for training, reduce `--batch-size` and increase
`--gradient-accumulation-steps` by the corresponding factor. A model closer to
1.5B may require batch size 2–4 or quantized loading, which this minimal
repository does not currently add.

## Quick checks

The smoke tests generate tiny temporary splits and check answer verification,
nested training sets, held-out templates, and required fields. They do not
download or train a language model.

```bash
pytest
```

To inspect all script options:

```bash
python scripts/generate_data.py --help
python scripts/generate_data_v2.py --help
python scripts/train_baseline.py --help
python scripts/train_structure.py --help
python scripts/train_oracle.py --help
python scripts/train_tool_routing.py --help
python scripts/evaluate.py --help
python scripts/evaluate_hybrid.py --help
```

## Interpreting the result

The key comparison is the structure-aware versus baseline answer accuracy at
each training-set size, especially on `test_unseen.jsonl`. A useful signal for
the hypothesis would be a larger advantage at 50 or 100 examples that narrows
as more examples are added. Structure and tool accuracy help distinguish
whether gains come with correct intermediate categorization rather than only a
different answer format.

### Oracle diagnostic

The oracle condition separates label recognition from symbolic execution. If
gold structure/tool labels substantially improve answer accuracy, the
intermediate representation is useful, while the structure-aware model may be
failing to predict or reliably use it. If oracle answer accuracy also remains
near floor, symbolic execution or model capacity is the more likely dominant
bottleneck.

The conditional metrics provide a second view: they test whether the
structure-aware model answers correctly specifically on examples where its
structure and/or tool prediction is correct. The evaluator also warns when the
two labels form a one-to-one mapping, because they are then effectively
redundant auxiliary targets in this dataset. This is diagnostic only; it does
not change the taxonomy.

If all answer-generation conditions remain at floor, this experiment by itself
does not prove or disprove the broader structure-supervision hypothesis.

Because the dataset is synthetic and the model is small, results should be
treated as evidence about this controlled setup only. Run multiple data and
training seeds before drawing a strong conclusion. This proof of concept also
matches runs by number of examples and epochs, not by the number of supervised
target tokens; the structured target is necessarily longer than the baseline
target.

## Repository layout

```text
src/math_structure/data.py  synthetic examples and SymPy verification
src/math_structure/data_v2.py  nonredundant taxonomy-v2 generation
src/math_structure/tools.py deterministic taxonomy-v2 plugin registry
scripts/generate_data.py    JSONL split generation
scripts/generate_data_v2.py taxonomy-v2 split generation
scripts/train_common.py     shared prompting and LoRA training helpers
scripts/train_baseline.py   answer-only training
scripts/train_structure.py  structure/tool/answer training
scripts/train_oracle.py     answer training with gold labels in the prompt
scripts/train_tool_routing.py structure/tool/argument routing training
scripts/evaluate.py         exact, conditional, mapping, and CSV evaluation
scripts/evaluate_hybrid.py  direct-versus-plugin evaluation and diagnostics
tests/                      fast data-generation smoke tests
```
