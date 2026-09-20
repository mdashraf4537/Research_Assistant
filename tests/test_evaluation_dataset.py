from app.evaluation.dataset import load_ground_truth


def test_load_ground_truth_accepts_pretty_printed_json_objects(tmp_path) -> None:
    dataset = tmp_path / "ground_truth.jsonl"
    dataset.write_text(
        '{\n  "user_input": "Question one", "reference": "Answer one"\n}\n'
        '{\n  "user_input": "Question two", "reference": "Answer two", '
        '"reference_contexts": ["Evidence"]\n}\n',
        encoding="utf-8",
    )

    cases = load_ground_truth(dataset)

    assert len(cases) == 2
    assert cases[1].reference_contexts == ["Evidence"]
