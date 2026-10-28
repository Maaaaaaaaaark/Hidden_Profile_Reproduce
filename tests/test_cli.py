from __future__ import annotations

import json

import hiddenbench.cli as cli


class FakeModel:
    model_name = "fake"

    def chat(self, messages):
        return "I am sharing my relevant facts."

    def chat_json(self, messages, schema):
        answer = schema["properties"]["vote"]["enum"][0]
        return {"vote": answer, "rationale": "The fake model picks a valid answer."}


def test_cli_inspect(capsys):
    exit_code = cli.main(["inspect", "--benchmark", "data/benchmark_short.json", "--task-id", "1"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "evacuation_west_city" in captured.out


def test_cli_eval_with_mocked_model(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "build_model_client", lambda **kwargs: FakeModel())
    output = tmp_path / "result.json"
    exit_code = cli.main(
        [
            "eval",
            "--benchmark",
            "data/benchmark_short.json",
            "--profile",
            "hidden",
            "--rounds",
            "1",
            "--duplications",
            "1",
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    data = json.loads(output.read_text())
    assert len(data["runs"]) == 3
    assert all(run["duplication_index"] == 0 for run in data["runs"])


def test_cli_eval_resume_skips_completed_runs(monkeypatch, tmp_path):
    output = tmp_path / "result.json"
    monkeypatch.setattr(cli, "build_model_client", lambda **kwargs: FakeModel())
    arguments = [
        "eval",
        "--benchmark",
        "data/benchmark_short.json",
        "--profile",
        "hidden",
        "--model",
        "fake",
        "--rounds",
        "1",
        "--duplications",
        "1",
        "--seed",
        "2026",
        "--resume",
        "--output",
        str(output),
    ]
    assert cli.main(arguments) == 0

    class ExplodingModel:
        model_name = "fake"

        def chat(self, messages):
            raise AssertionError("A completed run should not execute again")

        def chat_json(self, messages, schema):
            raise AssertionError("A completed run should not execute again")

    monkeypatch.setattr(cli, "build_model_client", lambda **kwargs: ExplodingModel())
    assert cli.main(arguments) == 0
    assert len(json.loads(output.read_text())["runs"]) == 3


def test_cli_score(tmp_path, capsys):
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "metadata": {"profile": "hidden", "model": "fake"},
                "runs": [
                    {
                        "task_id": 1,
                        "scenario": "evacuation_west_city",
                        "initial_votes": [{"vote": "West City"}],
                        "final_votes": [{"vote": "West City"}],
                    }
                ],
            }
        )
    )
    exit_code = cli.main(["score", str(result_path), "--benchmark", "data/benchmark_short.json"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Average rule" in captured.out


def test_cli_score_multiple_files(tmp_path, capsys):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    payload = {
        "metadata": {"profile": "hidden", "model": "fake"},
        "runs": [
            {
                "task_id": 1,
                "scenario": "evacuation_west_city",
                "initial_votes": [{"vote": "West City"}],
                "final_votes": [{"vote": "West City"}],
            }
        ],
    }
    first.write_text(json.dumps(payload))
    second.write_text(json.dumps(payload))
    exit_code = cli.main(["score", str(first), str(second), "--benchmark", "data/benchmark_short.json"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Runs: 2 across 1 tasks" in captured.out
