# Paper Result Validation

This check validates the curated public result JSON files against the paper's Figure 3 metrics using the `hiddenbench score` implementation.

The hidden-profile paper results are stored in two legacy files per model:

- `results/paper/hidden_manual_adapted/hidden_manual_adapted_<model>.json`
- `results/paper/hidden_generated/hidden_generated_<model>.json`

The full-profile reference is:

- `results/paper/full_profile/full_profile_<model>.json`

Example:

```bash
hiddenbench score \
  results/paper/hidden_manual_adapted/hidden_manual_adapted_gpt-4.1.json \
  results/paper/hidden_generated/hidden_generated_gpt-4.1.json \
  --benchmark data/benchmark.json \
  --full-results results/paper/full_profile/full_profile_gpt-4.1.json
```

## Validation Table

The table compares recomputed metrics with the paper's 3-decimal Figure 3 values.

| Model | Hidden pre | Hidden post | Full pre | Improvement | Paper improvement | Delta | Gap | Paper gap | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt-4o | 0.113 | 0.257 | 0.903 | 0.144 | 0.144 | +0.000 | -0.646 | -0.646 | +0.000 |
| gpt-4.1-nano | 0.156 | 0.240 | 0.562 | 0.084 | 0.084 | -0.000 | -0.322 | -0.322 | +0.000 |
| gpt-4.1-mini | 0.098 | 0.381 | 0.886 | 0.283 | 0.283 | -0.000 | -0.505 | -0.505 | +0.000 |
| gpt-4.1 | 0.082 | 0.342 | 0.969 | 0.260 | 0.260 | +0.000 | -0.627 | -0.627 | -0.000 |
| gpt-5-nano-minimal | 0.106 | 0.102 | 0.435 | -0.004 | -0.004 | -0.000 | -0.333 | -0.333 | +0.000 |
| gpt-5-mini-minimal | 0.108 | 0.223 | 0.807 | 0.114 | 0.114 | +0.000 | -0.585 | -0.585 | +0.000 |
| gpt-5-minimal | 0.121 | 0.186 | 0.936 | 0.064 | 0.064 | +0.000 | -0.750 | -0.750 | -0.000 |
| gpt-5-medium | 0.162 | 0.280 | 0.950 | 0.118 | 0.118 | +0.000 | -0.670 | -0.670 | +0.000 |
| gemini-2.5-flash-lite | 0.137 | 0.394 | 0.789 | 0.257 | 0.257 | +0.000 | -0.395 | -0.395 | +0.000 |
| gemini-2.5-flash | 0.201 | 0.550 | 0.972 | 0.349 | 0.349 | +0.000 | -0.422 | -0.422 | +0.000 |
| gemini-2.5-pro | 0.217 | 0.671 | 0.981 | 0.454 | 0.454 | +0.000 | -0.310 | -0.310 | -0.000 |
| qwen3-8b | 0.139 | 0.196 | 0.587 | 0.056 | 0.056 | +0.000 | -0.391 | -0.391 | -0.000 |
| qwen3-14b | 0.144 | 0.194 | 0.771 | 0.049 | 0.049 | +0.000 | -0.577 | -0.577 | -0.000 |
| qwen3-32b | 0.132 | 0.191 | 0.824 | 0.058 | 0.058 | +0.000 | -0.634 | -0.634 | +0.000 |
| qwen3-235b-a22b | 0.130 | 0.227 | 0.826 | 0.097 | 0.097 | -0.000 | -0.599 | -0.599 | +0.000 |
| llama-4-scout | 0.125 | 0.309 | 0.692 | 0.184 | 0.184 | -0.000 | -0.383 | -0.383 | -0.000 |
| llama-4 | 0.150 | 0.382 | 0.826 | 0.232 | 0.232 | +0.000 | -0.444 | -0.444 | +0.000 |

Maximum absolute delta against the paper's rounded values:

- Improvement: `0.000487`
- Gap to full profile: `0.000487`

These differences are consistent with comparing full-precision JSON-derived values to 3-decimal paper reporting.
