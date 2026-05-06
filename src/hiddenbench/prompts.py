from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptSet:
    system_prompt: str
    first_user_prompt: str
    user_prompt: str
    first_vote_prompt: str
    vote_prompt: str
    generation_prompt: str | None = None


PROMPT_FILES = {
    "system_prompt": "system_prompt.txt",
    "first_user_prompt": "first_user_prompt.txt",
    "user_prompt": "user_prompt.txt",
    "first_vote_prompt": "first_vote_prompt.txt",
    "vote_prompt": "vote.txt",
    "generation_prompt": "generate.txt",
}


def default_prompt_dir() -> Path:
    bundled = Path(__file__).resolve().parent / "prompts"
    if bundled.exists():
        return bundled
    package_root = Path(__file__).resolve().parents[2]
    return package_root / "prompts"


def load_prompts(path: str | Path | None = None) -> PromptSet:
    prompt_dir = Path(path) if path is not None else default_prompt_dir()
    values: dict[str, str | None] = {}
    for key, filename in PROMPT_FILES.items():
        file_path = prompt_dir / filename
        if key == "generation_prompt" and not file_path.exists():
            values[key] = None
            continue
        values[key] = file_path.read_text(encoding="utf-8")
    return PromptSet(**values)


def render_prompt(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"%{key}%", value)
    return rendered
