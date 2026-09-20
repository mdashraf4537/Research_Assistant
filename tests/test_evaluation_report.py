from app.evaluation.runner import (
    EvaluationCaseResult,
    EvaluationReport,
    ground_truth_sha256,
    load_cached_report,
    save_report,
)


def test_save_report_writes_machine_readable_json(tmp_path) -> None:
    report = EvaluationReport(
        created_at="2026-08-19T00:00:00+00:00",
        dataset_size=1,
        averages={"faithfulness": 1.0},
        targets={"faithfulness": 0.85},
        passed=True,
        cases=[
            EvaluationCaseResult(
                user_input="Question",
                response="Answer",
                reference="Reference",
                retrieved_contexts=["Evidence"],
                metrics={"faithfulness": 1.0},
                reasons={"faithfulness": "supported"},
            )
        ],
    )
    path = save_report(report, tmp_path)

    assert path.exists()
    assert '"passed": true' in path.read_text(encoding="utf-8")


def test_load_cached_report_reuses_only_an_identical_dataset(tmp_path) -> None:
    dataset = tmp_path / "ground_truth.json"
    dataset.write_text('{"user_input":"Q"}', encoding="utf-8")
    fingerprint = ground_truth_sha256(dataset)
    report = EvaluationReport(
        created_at="2026-09-08T00:00:00+00:00",
        dataset_size=1,
        averages={"faithfulness": 1.0},
        targets={"faithfulness": 0.85},
        passed=True,
        cases=[],
        dataset_sha256=fingerprint,
    )
    saved = save_report(report, tmp_path)

    cached = load_cached_report(tmp_path, fingerprint)

    assert cached is not None
    assert cached[1] == saved
    assert load_cached_report(tmp_path, "changed-dataset") is None
