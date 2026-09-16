# Fourth-round hybrid experiment summary

Model: `Qwen/Qwen2.5-0.5B-Instruct` with the same five-epoch LoRA setup, seed,
and taxonomy-v2 splits as the third round. Direct scores are reused from the
previous per-example Qwen-v2 predictions after ID/problem/answer validation.
The routing model predicts structure, tool, and JSON arguments; it never emits
the final answer as its supervised target.

| train | split | baseline direct | structure direct | predicted plugin | oracle plugin | structure | tool | semantic args | plugin success |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | seen | 4.2% | 7.5% | 10.0% | 100.0% | 13.3% | 51.7% | 7.5% | 17.5% |
| 50 | unseen | 6.7% | 0.0% | 3.3% | 100.0% | 0.0% | 56.7% | 0.0% | 9.2% |
| 100 | seen | 10.8% | 11.7% | 62.5% | 100.0% | 94.2% | 99.2% | 61.7% | 70.8% |
| 100 | unseen | 3.3% | 3.3% | 23.3% | 100.0% | 36.7% | 70.8% | 15.0% | 45.8% |
| 200 | seen | 15.0% | 15.8% | 88.3% | 100.0% | 100.0% | 100.0% | 88.3% | 99.2% |
| 200 | unseen | 7.5% | 8.3% | 32.5% | 100.0% | 51.7% | 79.2% | 31.7% | 45.8% |
| 500 | seen | 13.3% | 16.7% | 98.3% | 100.0% | 100.0% | 100.0% | 98.3% | 99.2% |
| 500 | unseen | 9.2% | 10.8% | 36.7% | 100.0% | 50.8% | 64.2% | 30.8% | 51.7% |

At 500 examples, the predicted plugin system exceeds structure-aware direct
generation by 81.7 percentage points on seen templates and 25.8 points on
unseen templates. This is the requested `predicted_plugin >> direct_generation`
pattern for this controlled run. It supports the narrow claim that the small
model is more useful as a router into deterministic algebra than as the algebra
executor itself; it is not evidence of pure end-to-end neural reasoning.

Execution is not the residual bottleneck here. Oracle execution is perfect,
and every prediction with both the correct tool and semantically equivalent
arguments produces the correct final answer. At 500/unseen, the 76 failed
examples break down as 36 `wrong_structure`, 23 `wrong_tool`, 12
`invalid_arguments`, and 5 `valid_but_wrong_arguments`; there are no
deterministic execution or plugin-answer-mismatch failures. The comparatively
low unseen score therefore points to surface-template generalization and
argument extraction/routing as the remaining limitations.

The exact unrounded metrics are in `metrics.csv`; all 960 model completions,
parsed fields, plugin outputs, and primary error categories are in
`predictions.csv`.
