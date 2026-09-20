import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GroundTruthCase:
    user_input: str
    reference: str
    reference_contexts: list[str]


def load_ground_truth(path: str | Path) -> list[GroundTruthCase]:
    raw = Path(path).read_text(encoding="utf-8-sig").strip()

    if not raw:
        raise ValueError("The ground-truth dataset is empty.")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # fallback for the older JSONL-style format
        decoder = json.JSONDecoder()
        position = 0
        items: list[dict[str, object]] = []
        while position < len(raw):
            while position < len(raw) and raw[position].isspace():
                position += 1
            if position >= len(raw):
                break
            item, position = decoder.raw_decode(raw, position)
            if not isinstance(item, dict):
                raise ValueError("Each ground-truth record must be a JSON object.")
            items.append(item)
    else:
        if isinstance(parsed, dict):
            items = [parsed]
        elif isinstance(parsed, list):
            items = parsed
        else:
            raise ValueError("Ground-truth data must be a JSON object or a JSON array.")
    
    cases: list[GroundTruthCase] = []
    for number, item in enumerate(items, start=1):
        user_input = item.get("user_input")
        reference = item.get("reference")
        reference_contexts = item.get("reference_contexts", [])
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError(f"Record {number}: user_input must be a non-empty string.")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(f"Record {number}: reference must be a non-empty string.")
        if not isinstance(reference_contexts, list) or not all(
            isinstance(context, str) for context in reference_contexts
        ):
            raise ValueError(f"Record {number}: reference_contexts must be a list of strings.")
        cases.append(
            GroundTruthCase(
                user_input=user_input.strip(),
                reference=reference.strip(),
                reference_contexts=[
                    context.strip() for context in reference_contexts if context.strip()
                ],
            )
        )

    if not cases:
        raise ValueError("The ground-truth dataset is empty.")
    return cases