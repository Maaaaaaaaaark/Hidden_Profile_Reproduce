---
license: mit
pretty_name: HiddenBench Result Files
tags:
  - multi-agent
  - large-language-models
  - benchmark
  - collective-reasoning
  - hidden-profile
---

# HiddenBench Result Files

Curated discussion and vote JSON files from the HiddenBench experiments. File contents are copied without modification; only names and folder placement were changed for clarity.

## Contents

- `MANIFEST.json`: model names, run counts, benchmark split, and profile metadata.
- `paper/hidden_manual_adapted/`: hidden-profile runs for the 3 manually designed plus 5 adapted tasks.
- `paper/hidden_generated/`: hidden-profile runs for the 57 automatically generated tasks.
- `paper/full_profile/`: full-profile runs for all 65 tasks.
- `short_and_ablations/`: short benchmark and diagnostic protocol files.
- `hiddenbench-results-json.zip`: compressed archive containing the same files under a local `results/` folder.

## Analyze

```bash
hiddenbench score \
  results/paper/hidden_manual_adapted/hidden_manual_adapted_gpt-4.1.json \
  results/paper/hidden_generated/hidden_generated_gpt-4.1.json \
  --benchmark data/benchmark.json \
  --full-results results/paper/full_profile/full_profile_gpt-4.1.json
```
