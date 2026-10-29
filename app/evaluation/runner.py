import asyncio
import hashlib
import inspect
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

from app.answering.service import ResearchAnswerService
from app.core.config import Settings
from app.evaluation.dataset import GroundTruthCase, load_ground_truth


@dataclass(frozen=True)
class EvaluationCaseResult:
    user_input: str
    response: str
    reference: str
    retrieved_contexts: list[str]
    metrics: dict[str, float]
    reasons: dict[str, str | None]


@dataclass(frozen=True)
class EvaluationReport:
    created_at: str
    dataset_size: int
    averages: dict[str, float]
    targets: dict[str, float]
    passed: bool
    cases: list[EvaluationCaseResult]
    dataset_sha256: str = ""


class RagasEvaluationRunner:
    """Run current Ragas collections metrics over real cited-RAG responses."""

    def __init__(self, answer_service: ResearchAnswerService, settings: Settings) -> None:
        self._answer_service = answer_service
        self._settings = settings
    
    # async def run(self, dataset_path: str | Path) -> EvaluationReport:
    #     cases = load_ground_truth(dataset_path)
    #     metrics = self._build_metrics()
    #     results = [await self._evaluate_case(case, metrics) for case in cases]
    #     averages = {
    #         name: round(fmean(result.metrics[name] for result in results), 4) for name in metrics
    #     }
    async def run(self, dataset_path: str | Path) -> EvaluationReport:
         cases = load_ground_truth(dataset_path)
         metrics = self._build_metrics()
         results = []
         for case in cases:
           result = await self._evaluate_case(case, metrics)
           results.append(result)
           await asyncio.sleep(self._settings.evaluation_throttle_seconds)  # throttle between cases
         averages = {
           name: round(fmean(result.metrics[name] for result in results), 4) for name in metrics
         }
         targets = {
            "faithfulness": 0.7,
            "factual_correctness": 0.7,
            "context_precision": 0.7,
            "context_recall": 0.7,
        }
         passed = all(averages[name] >= target for name, target in targets.items())
         return EvaluationReport(
            created_at=datetime.now(UTC).isoformat(),
            dataset_size=len(results),
            averages=averages,
            targets=targets,
            passed=passed,
            cases=results,
            dataset_sha256=ground_truth_sha256(dataset_path),
        )

    def _build_metrics(self) -> dict[str, Any]:
        """Lazy imports keep regular API startup independent of Ragas dependencies."""
        from openai import AsyncOpenAI

        from app.evaluation.ragas_compat import install_legacy_vertexai_import

        install_legacy_vertexai_import()
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            ContextPrecisionWithReference,
            ContextRecall,
            FactualCorrectness,
            Faithfulness,
        )

        if self._settings.deepseek_api_key is None:
            raise RuntimeError("DEEPSEEK_API_KEY is required to run Ragas evaluation.")
        client = AsyncOpenAI(
            api_key=self._settings.deepseek_api_key.get_secret_value(),
            base_url="https://api.deepseek.com",
            timeout=60.0,
            max_retries=2,
        )
        evaluator_llm = llm_factory(self._settings.evaluation_deepseek_model, client=client)
        # Patch model_args after creation — adapter hardcodes InstructorModelArgs()
        # and _map_openai_params calls .copy() which returns a Pydantic model, not a dict.
        # Override _map_provider_params to return a plain dict with increased max_tokens.
        def _patched_map_provider_params() -> dict:
            return {"max_tokens": 8192, "temperature": 0.01, "top_p": 0.1}
        evaluator_llm._map_provider_params = _patched_map_provider_params
        return {
            "faithfulness": Faithfulness(llm=evaluator_llm),
            "factual_correctness": FactualCorrectness(llm=evaluator_llm),
            "context_precision": ContextPrecisionWithReference(llm=evaluator_llm),
            "context_recall": ContextRecall(llm=evaluator_llm),
        }
    async def _score_metric(self, metric: Any, metric_inputs: dict) -> Any:
        sig = inspect.signature(metric.ascore)
        accepted = set(sig.parameters.keys())
        filtered_inputs = {k: v for k, v in metric_inputs.items() if k in accepted}
        last_exc: Exception | None = None
        for attempt in range(6):
            try:
                return await metric.ascore(**filtered_inputs)
            except Exception as exc:
                last_exc = exc
                if "429" in str(exc) or "rate_limit" in str(exc).lower():
                    wait = 20 * (attempt + 1)
                    await asyncio.sleep(wait)
                else:
                    raise
        raise RuntimeError(f"Exhausted retries due to rate limits: {last_exc}")
    
    async def _evaluate_case(
        self, case: GroundTruthCase, metrics: dict[str, Any]
    ) -> EvaluationCaseResult:
        answer = await self._answer_service.answer(case.user_input)
        contexts = [citation.excerpt for citation in answer.citations]
        metric_inputs = {
            "user_input": case.user_input,
            "response": answer.answer,
            "reference": case.reference,
            "retrieved_contexts": [c[:2500] for c in contexts],
            "reference_contexts": case.reference_contexts,
        }
        metric_results = {}
        for name, metric in metrics.items():
            metric_results[name] = await self._score_metric(metric, metric_inputs)
            if self._settings.evaluation_throttle_seconds > 0:
                await asyncio.sleep(self._settings.evaluation_throttle_seconds)
        
        return EvaluationCaseResult(
            user_input=case.user_input,
            response=answer.answer,
            reference=case.reference,
            retrieved_contexts=contexts,
            metrics={name: float(result.value) for name, result in metric_results.items()},
            reasons={
                name: getattr(result, "reason", None) for name, result in metric_results.items()
            },
        )


def save_report(report: EvaluationReport, output_directory: str | Path) -> Path:
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"ragas-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def ground_truth_sha256(dataset_path: str | Path) -> str:
    """Fingerprint the exact dataset bytes used to produce an evaluation report."""
    return hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()


def load_cached_report(
    output_directory: str | Path, dataset_sha256: str
) -> tuple[EvaluationReport, Path] | None:
    """Return the newest completed report for an unchanged ground-truth dataset."""
    reports = sorted(
        Path(output_directory).glob("ragas-*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in reports:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("dataset_sha256") != dataset_sha256:
                continue
            return (
                EvaluationReport(
                    created_at=payload["created_at"],
                    dataset_size=payload["dataset_size"],
                    averages=payload["averages"],
                    targets=payload["targets"],
                    passed=payload["passed"],
                    cases=[EvaluationCaseResult(**case) for case in payload["cases"]],
                    dataset_sha256=payload["dataset_sha256"],
                ),
                path,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return None


def main() -> None:
    from app.api.routes import get_answer_service
    from app.core.config import get_settings

    settings = get_settings()
    dataset_path = (
        sys.argv[1] if len(sys.argv) > 1 else "data/evaluations/ground_truth.json"
    )
    dataset_sha256 = ground_truth_sha256(dataset_path)
    cached = load_cached_report(settings.evaluation_results_directory, dataset_sha256)
    if cached is not None:
        report, path = cached
        cache_hit = True
    else:
        report = asyncio.run(
            RagasEvaluationRunner(get_answer_service(), settings).run(dataset_path)
        )
        path = save_report(report, settings.evaluation_results_directory)
        cache_hit = False
    print(
        json.dumps(
            {
                "passed": report.passed,
                "averages": report.averages,
                "report": str(path),
                "cached": cache_hit,
            }
        )
    )


if __name__ == "__main__":
    main()
