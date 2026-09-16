# Second-round diagnostic summary

Run date: 2026-09-16

## Configuration

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB)
- Runtime: Python 3.11.15, PyTorch 2.5.1+cu121, Transformers 4.57.3,
  PEFT 0.20.0, Accelerate 1.14.0
- LoRA: rank 8, alpha 16, dropout 0.05, `q_proj` and `v_proj`
- Training: 5 epochs, batch size 8, learning rate 2e-4, BF16, seed 17
- Data: seed 42; 120 seen-template and 120 unseen-template test examples

## Exact answer accuracy

| Train size | Split | Baseline | Structure-aware | Oracle labels |
|---:|:---|---:|---:|---:|
| 50  | seen   | 1.67% | 0.83% | 2.50% |
| 50  | unseen | 0.83% | 0.83% | 0.83% |
| 100 | seen   | 1.67% | 1.67% | 2.50% |
| 100 | unseen | 0.00% | 0.83% | 0.00% |
| 200 | seen   | 2.50% | 2.50% | 0.83% |
| 200 | unseen | 0.00% | 0.00% | 0.83% |
| 500 | seen   | 1.67% | 1.67% | 2.50% |
| 500 | unseen | 2.50% | 0.83% | 3.33% |

Oracle labels do not produce a substantial or consistent improvement. All
three answer-generation conditions remain close to floor.

## Conditional execution diagnosis

At 200 and 500 examples the structure-aware model predicts both labels
correctly on every seen-template example, but its answer accuracy on that
subset is still only 2.50% and 1.67%, respectively. On unseen templates:

| Train size | Both labels correct (n) | Answer accuracy given both correct | Answer accuracy when either is wrong |
|---:|---:|---:|---:|
| 50  | 0  | n/a | 0.83% |
| 100 | 4  | 0.00% | 0.86% |
| 200 | 55 | 0.00% | 0.00% |
| 500 | 87 | 1.15% | 0.00% |

The full structure-conditioned and tool-conditioned results are in
`conditional_metrics.csv` and the combined wide table is in `comparison.csv`.

## Label mapping

The generated dataset has a one-to-one mapping in both directions: each of the
six structures has exactly one tool, and each tool belongs to exactly one
structure. The auxiliary labels are therefore redundant for this taxonomy.
The complete mapping is saved in `label_mapping.json`.

## Interpretation

For this 135M model and one training seed, correct intermediate label
recognition is not the main bottleneck. Providing gold labels does not lift
answer accuracy, and answer accuracy stays near floor even when the
structure-aware model predicts both labels correctly. Symbolic execution or
model capacity is therefore the more likely dominant bottleneck in this
controlled run.

This result does not prove or disprove the broader structure-supervision
hypothesis: the answer-generation conditions are all too close to floor for a
clean sample-efficiency comparison. A larger causal model or an executable
symbolic-tool intervention is the natural next diagnostic.
