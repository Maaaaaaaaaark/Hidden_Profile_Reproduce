from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def load_meta(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    conditions = data.get("conditions", []) if isinstance(data, dict) else []
    return {
        "conditions": len(conditions),
        "runs": sum(len(c.get("runs", [])) for c in conditions if isinstance(c, dict)),
        "models": sorted({c.get("model") for c in conditions if isinstance(c, dict) and c.get("model")}),
    }


def model_from_name(path: Path) -> str:
    name = path.stem
    return name.removeprefix("results_")


def copy_entry(
    manifest: list[dict],
    src: Path,
    dst_dir: Path,
    prefix: str,
    benchmark: str,
    profile: str,
    notes: str,
    results_root: Path,
) -> None:
    model = model_from_name(src)
    dst = dst_dir / f"{prefix}_{model}.json"
    shutil.copy2(src, dst)
    meta = load_meta(dst)
    manifest.append(
        {
            "path": str(dst.relative_to(results_root)),
            "original_filename": src.name,
            "benchmark": benchmark,
            "profile": profile,
            "models_in_file": meta["models"],
            "conditions": meta["conditions"],
            "runs": meta["runs"],
            "bytes": dst.stat().st_size,
            "notes": notes,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Curate legacy HiddenBench result JSON files into a public bundle.")
    parser.add_argument("--source", required=True, help="Path to the legacy data/results directory.")
    parser.add_argument("--output", default="results", help="Output folder for curated files.")
    parser.add_argument("--zip", default="hiddenbench-results-json.zip", help="ZIP archive path.")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)
    for subdir in [
        "paper/hidden_manual_adapted",
        "paper/hidden_generated",
        "paper/full_profile",
        "short_and_ablations",
    ]:
        (output / subdir).mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []

    for path in sorted(source.glob("results_*.json")):
        if "_tmp_" in path.name:
            continue
        if any(
            token in path.name
            for token in ["hidden_short", "complete_short", "asymmetric", "zero_shot", "spectrum", "dice", "free_", "diverse"]
        ):
            continue
        if load_meta(path)["runs"] == 80:
            copy_entry(
                manifest,
                path,
                output / "paper/hidden_manual_adapted",
                "hidden_manual_adapted",
                "8 manual/adapted tasks",
                "hidden",
                "Hidden-profile runs for the 3 manually designed plus 5 adapted tasks.",
                output,
            )

    for path in sorted((source / "generated").glob("results_*.json")):
        if "_tmp_" not in path.name and load_meta(path)["runs"] == 570:
            copy_entry(
                manifest,
                path,
                output / "paper/hidden_generated",
                "hidden_generated",
                "57 generated tasks",
                "hidden",
                "Hidden-profile runs for automatically generated HiddenBench tasks.",
                output,
            )

    for path in sorted((source / "complete").glob("results_*.json")):
        if "_tmp_" not in path.name and load_meta(path)["runs"] == 650:
            copy_entry(
                manifest,
                path,
                output / "paper/full_profile",
                "full_profile",
                "65 HiddenBench tasks",
                "full",
                "Full-profile runs for all 65 HiddenBench tasks.",
                output,
            )

    diagnostic_files = (
        sorted(source.glob("results_hidden_short*.json"))
        + sorted(source.glob("results_complete_short.json"))
        + sorted(source.glob("results_zero_shot_cot.json"))
        + sorted(source.glob("results_asymmetric.json"))
    )
    for path in diagnostic_files:
        if "_tmp_" not in path.name and path.exists():
            copy_entry(
                manifest,
                path,
                output / "short_and_ablations",
                path.stem,
                "short/diagnostic subset",
                "mixed/legacy",
                "Short benchmark or diagnostic protocol result file.",
                output,
            )

    manifest.sort(key=lambda row: row["path"])
    (output / "MANIFEST.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "description": (
                    "Curated HiddenBench discussion and vote JSON result files. Result file contents are copied "
                    "without modification from the research workspace; only filenames and folder placement were "
                    "changed for clarity."
                ),
                "total_files": len(manifest),
                "files": manifest,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# HiddenBench Result Files\n\n"
        "Curated discussion and vote JSON files from the HiddenBench experiments. File contents are copied without "
        "modification; only names and folder placement were changed for clarity.\n\n"
        "See `MANIFEST.json` for model names, run counts, benchmark split, and profile metadata.\n",
        encoding="utf-8",
    )

    zip_path = Path(args.zip)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output.parent))

    print(f"Wrote {len(manifest)} files to {output}")
    print(f"Wrote archive to {zip_path} ({zip_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
