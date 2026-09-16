# Third-round experiment summary

Run date: 2026-09-16

## Shared configuration

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB)
- Training: 5 epochs, batch size 8, learning rate 2e-4, BF16, seed 17
- LoRA: rank 8, alpha 16, dropout 0.05, automatically detected `q_proj` and
  `v_proj`
- Memory: gradient checkpointing enabled
- Data seed: 42
- Train sizes: 50, 100, 200, 500
- Test data: 120 seen-template and 120 unseen-template examples per experiment

## Experiment A — larger model, original dataset

This experiment changes model capacity only. The original dataset, splits, and
one-to-one structure/tool taxonomy are unchanged.

| Train | Split | Baseline | Structure-aware | Oracle | Structure acc. | Tool acc. | Answer given both labels correct |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 50 | seen | 6.67% | 10.00% | 5.83% | 73.33% | 54.17% | 13.33% |
| 50 | unseen | 4.17% | 7.50% | 6.67% | 40.83% | 48.33% | 2.70% |
| 100 | seen | 7.50% | 9.17% | 10.00% | 100.00% | 100.00% | 9.17% |
| 100 | unseen | 6.67% | 7.50% | 5.00% | 86.67% | 78.33% | 6.38% |
| 200 | seen | 11.67% | 12.50% | 12.50% | 100.00% | 100.00% | 12.50% |
| 200 | unseen | 9.17% | 6.67% | 11.67% | 91.67% | 83.33% | 7.00% |
| 500 | seen | 20.83% | 19.17% | 17.50% | 100.00% | 100.00% | 19.17% |
| 500 | unseen | 10.83% | 10.83% | 11.67% | 91.67% | 83.33% | 9.00% |

Answer accuracy is clearly above the 135M floor, so increasing capacity was a
useful diagnostic. Structure-aware training leads at 50 examples on both
splits and has smaller gains at 100 examples, but the advantage does not
persist at 200–500 examples. One seed is not enough to treat the early lead as
strong evidence of improved sample efficiency. Oracle labels again do not
produce a consistent advantage.

## Experiment B — larger model, taxonomy v2

This is a separate experiment with a different problem/taxonomy mix. It must
not be used as a controlled estimate of model-capacity effects.

| Train | Split | Baseline | Structure-aware | Oracle | Structure acc. | Tool acc. | Answer given both labels correct |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 50 | seen | 4.17% | 7.50% | 5.00% | 57.50% | 70.83% | 6.12% |
| 50 | unseen | 6.67% | 0.00% | 4.17% | 8.33% | 60.83% | 0.00% |
| 100 | seen | 10.83% | 11.67% | 5.00% | 99.17% | 98.33% | 11.02% |
| 100 | unseen | 3.33% | 3.33% | 5.83% | 65.00% | 80.00% | 5.56% |
| 200 | seen | 15.00% | 15.83% | 8.33% | 100.00% | 100.00% | 15.83% |
| 200 | unseen | 7.50% | 8.33% | 7.50% | 61.67% | 75.83% | 9.38% |
| 500 | seen | 13.33% | 16.67% | 17.50% | 100.00% | 100.00% | 16.67% |
| 500 | unseen | 9.17% | 10.83% | 8.33% | 71.67% | 80.83% | 11.27% |

The taxonomy is genuinely nonredundant: all six structures have two canonical
tools across the dataset, and expansion, factorization, and substitution each
apply to multiple structures. Recognition, method selection, and execution are
therefore distinguishable.

Structure-aware answer accuracy is slightly higher on most seen-template
points and at 200–500 unseen examples, but it collapses on the 50-example
unseen split and ties baseline at 100. This run does not show a consistent
small-data structural-generalization advantage. It is neither at floor nor at
ceiling, so multiple training/data seeds are the next useful test.

## Outputs

- Experiment A: `results_qwen_original/`
- Experiment B: `results_qwen_v2/`
- Both directories contain the wide comparison, long metrics, conditional
  execution metrics, label mapping, structure/tool confusion counts, and all
  per-example predictions.
