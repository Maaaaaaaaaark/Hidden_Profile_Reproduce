# Public Result Files

The curated result bundle contains generated discussion and vote JSON files from the HiddenBench experiments. The JSON contents are copied without modification from the research workspace; only folder placement and filenames are clarified.

Public dataset: [YuxuanLi1225/HiddenBench-results](https://huggingface.co/datasets/YuxuanLi1225/HiddenBench-results)

## Bundle Contents

- `MANIFEST.json`: file-level metadata, including model names, run counts, benchmark split, and profile.
- `paper/hidden_manual_adapted/`: hidden-profile runs for the 3 manually designed plus 5 adapted tasks.
- `paper/hidden_generated/`: hidden-profile runs for the 57 automatically generated tasks.
- `paper/full_profile/`: full-profile runs for all 65 tasks.
- `short_and_ablations/`: short benchmark and diagnostic protocol files.
- `hiddenbench-results-json.zip`: compressed archive containing the same files under a local `results/` folder.

The primary hidden-profile paper results are split into `hidden_manual_adapted` and `hidden_generated` because that is how the original runs were stored. Pass both files to `hiddenbench score` to evaluate all 65 tasks.

## Download

```bash
pip install -U huggingface_hub
hf download YuxuanLi1225/HiddenBench-results hiddenbench-results-json.zip \
  --repo-type dataset \
  --local-dir .
unzip hiddenbench-results-json.zip
```

To download the folder instead of the ZIP:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="YuxuanLi1225/HiddenBench-results",
    repo_type="dataset",
    allow_patterns=["MANIFEST.json", "README.md", "paper/**", "short_and_ablations/**"],
    local_dir="results",
)
```

## Analyze

Score the GPT-4.1 hidden-profile result across all 65 tasks:

```bash
hiddenbench score \
  results/paper/hidden_manual_adapted/hidden_manual_adapted_gpt-4.1.json \
  results/paper/hidden_generated/hidden_generated_gpt-4.1.json \
  --benchmark data/benchmark.json \
  --full-results results/paper/full_profile/full_profile_gpt-4.1.json
```

Score a full-profile file:

```bash
hiddenbench score \
  results/paper/full_profile/full_profile_gemini-2.5-pro.json \
  --benchmark data/benchmark.json
```

The scorer supports both the legacy `conditions -> runs` schema used by these artifacts and the newer `metadata -> runs` schema produced by the open-source CLI.
