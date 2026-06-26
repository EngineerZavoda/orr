"""Scan and benchmark free OpenRouter chat models."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Mapping, Optional, Sequence

import requests
from dotenv import load_dotenv


MODELS_URL = "https://openrouter.ai/api/v1/models"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_PROMPT = "Say 'ok' in one word"


class ScannerError(RuntimeError):
    """An error that prevents the scanner from running."""


@dataclass(frozen=True)
class ModelInfo:
    id: str
    params: str = "N/A"


@dataclass(frozen=True)
class AttemptResult:
    status: str
    elapsed: Optional[float] = None
    detail: Optional[str] = None
    retry_after: Optional[str] = None


@dataclass
class ModelResult:
    model: ModelInfo
    attempts: List[AttemptResult] = field(default_factory=list)

    @property
    def successful_runs(self) -> int:
        return sum(attempt.status == "OK" for attempt in self.attempts)

    @property
    def median_latency(self) -> Optional[float]:
        values = [
            attempt.elapsed
            for attempt in self.attempts
            if attempt.status == "OK" and attempt.elapsed is not None
        ]
        return round(float(median(values)), 3) if values else None

    @property
    def status(self) -> str:
        if self.successful_runs:
            return "OK" if self.successful_runs == len(self.attempts) else "PARTIAL"
        return self.attempts[-1].status if self.attempts else "NOT_RUN"


def build_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def extract_params(model_id: str) -> str:
    """Extract the first parameter-like size such as 7B or 1.5B from an id."""
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)[bB](?![A-Za-z])", model_id)
    return f"{match.group(1)}B" if match else "N/A"


def _is_zero_price(value: Any) -> bool:
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def parse_free_models(payload: Any) -> List[ModelInfo]:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ScannerError("OpenRouter вернул некорректный список моделей.")

    models: List[ModelInfo] = []
    for raw_model in payload["data"]:
        if not isinstance(raw_model, dict):
            continue
        model_id = raw_model.get("id")
        pricing = raw_model.get("pricing")
        if (
            not isinstance(model_id, str)
            or not model_id.strip()
            or not isinstance(pricing, dict)
            or not _is_zero_price(pricing.get("prompt"))
            or not _is_zero_price(pricing.get("completion"))
        ):
            continue
        models.append(ModelInfo(id=model_id, params=extract_params(model_id)))
    return models


def get_free_models(
    session: requests.Session,
    headers: Mapping[str, str],
    timeout: float,
) -> List[ModelInfo]:
    try:
        response = session.get(MODELS_URL, headers=dict(headers), timeout=(5, timeout))
    except requests.Timeout as exc:
        raise ScannerError("Истекло время ожидания списка моделей.") from exc
    except requests.ConnectionError as exc:
        raise ScannerError("Не удалось подключиться к OpenRouter.") from exc
    except requests.RequestException as exc:
        raise ScannerError(f"Ошибка при получении моделей: {exc}") from exc

    if response.status_code in (401, 403):
        raise ScannerError("OpenRouter отклонил API-ключ или запретил доступ.")
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        suffix = f" Повторите через {retry_after} с." if retry_after else ""
        raise ScannerError(f"Превышен лимит OpenRouter.{suffix}")
    if response.status_code >= 500:
        raise ScannerError(f"OpenRouter временно недоступен (HTTP {response.status_code}).")
    if response.status_code != 200:
        raise ScannerError(f"Не удалось получить модели (HTTP {response.status_code}).")

    try:
        return parse_free_models(response.json())
    except ValueError as exc:
        raise ScannerError("OpenRouter вернул некорректный JSON со списком моделей.") from exc


def _response_detail(response: requests.Response) -> Optional[str]:
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"][:200]
    return None


def run_attempt(
    session: requests.Session,
    headers: Mapping[str, str],
    model_id: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> AttemptResult:
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    started = time.perf_counter()
    try:
        response = session.post(
            CHAT_URL,
            headers=dict(headers),
            json=payload,
            timeout=(5, timeout),
        )
        elapsed = round(time.perf_counter() - started, 3)
    except requests.Timeout:
        return AttemptResult("TIMEOUT")
    except requests.ConnectionError as exc:
        return AttemptResult("CONNECTION_ERROR", detail=str(exc)[:200])
    except requests.RequestException as exc:
        return AttemptResult("REQUEST_ERROR", detail=str(exc)[:200])

    detail = _response_detail(response)
    if response.status_code == 200:
        try:
            body = response.json()
        except ValueError:
            return AttemptResult("INVALID_JSON", elapsed, "Ответ не является JSON")
        if not isinstance(body, dict) or not isinstance(body.get("choices"), list):
            return AttemptResult("INVALID_JSON", elapsed, "В ответе отсутствует choices")
        return AttemptResult("OK", elapsed)
    if response.status_code in (401, 403):
        return AttemptResult("AUTH_ERROR", elapsed, detail)
    if response.status_code == 404:
        return AttemptResult("NOT_FOUND", elapsed, detail)
    if response.status_code == 429:
        return AttemptResult(
            "RATE_LIMITED",
            elapsed,
            detail,
            response.headers.get("Retry-After"),
        )
    if response.status_code >= 500:
        return AttemptResult("SERVER_ERROR", elapsed, detail)
    return AttemptResult(f"HTTP_{response.status_code}", elapsed, detail)


def benchmark_model(
    session: requests.Session,
    headers: Mapping[str, str],
    model: ModelInfo,
    runs: int,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> ModelResult:
    return ModelResult(
        model=model,
        attempts=[
            run_attempt(session, headers, model.id, prompt, max_tokens, timeout)
            for _ in range(runs)
        ],
    )


def benchmark_models(
    session: requests.Session,
    headers: Mapping[str, str],
    models: Sequence[ModelInfo],
    workers: int,
    runs: int,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> List[ModelResult]:
    results: List[ModelResult] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                benchmark_model,
                session,
                headers,
                model,
                runs,
                prompt,
                max_tokens,
                timeout,
            )
            for model in models
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def sort_results(results: Sequence[ModelResult], order: str) -> List[ModelResult]:
    if order == "latency":
        return sorted(
            results,
            key=lambda result: (
                result.median_latency is None,
                result.median_latency if result.median_latency is not None else float("inf"),
                result.model.id.lower(),
            ),
        )
    if order == "status":
        return sorted(results, key=lambda result: (result.status, result.model.id.lower()))
    return sorted(results, key=lambda result: result.model.id.lower())


def fastest_result(results: Sequence[ModelResult]) -> Optional[ModelResult]:
    successful = [result for result in results if result.median_latency is not None]
    return (
        min(successful, key=lambda result: (result.median_latency, result.model.id))
        if successful
        else None
    )


def print_results(results: Sequence[ModelResult], runs: int) -> None:
    model_width = max([len("Model"), *(len(result.model.id) for result in results)])
    header = (
        f"{'Model':{model_width}} | {'Params':8} | {'Status':16} | "
        f"{'Runs':7} | {'Median(s)':9}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        latency = f"{result.median_latency:.3f}" if result.median_latency is not None else "-"
        run_count = f"{result.successful_runs}/{runs}"
        print(
            f"{result.model.id:{model_width}} | {result.model.params:8} | "
            f"{result.status:16} | {run_count:7} | {latency:9}"
        )

    notes = []
    for result in results:
        if result.status in ("OK", "PARTIAL") or not result.attempts:
            continue
        attempt = result.attempts[-1]
        detail = attempt.detail
        if attempt.retry_after:
            detail = f"повторить через {attempt.retry_after} с" + (
                f"; {detail}" if detail else ""
            )
        if detail:
            notes.append((result.model.id, detail))
    if notes:
        print("\nПодробности ошибок:")
        for model_id, detail in notes:
            print(f"- {model_id}: {detail}")


def export_results(path: Path, results: Sequence[ModelResult]) -> None:
    payload = []
    for result in results:
        item = {
            "model": asdict(result.model),
            "status": result.status,
            "successful_runs": result.successful_runs,
            "median_latency": result.median_latency,
            "attempts": [asdict(attempt) for attempt in result.attempts],
        }
        payload.append(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Поиск и проверка бесплатных chat-моделей OpenRouter."
    )
    parser.add_argument("--workers", type=int, default=3, help="число параллельных моделей (3)")
    parser.add_argument("--timeout", type=float, default=15, help="таймаут чтения в секундах (15)")
    parser.add_argument("--runs", type=int, default=1, help="запросов на модель (1)")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="подстрока id модели; можно указать несколько раз",
    )
    parser.add_argument("--sort", choices=("latency", "name", "status"), default="latency")
    parser.add_argument("--json", type=Path, metavar="PATH", help="сохранить результаты в JSON")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="тестовый запрос")
    parser.add_argument("--max-tokens", type=int, default=5, help="максимум токенов ответа (5)")
    return parser


def _positive(parser: argparse.ArgumentParser, value: float, option: str) -> None:
    if not math.isfinite(value) or value <= 0:
        parser.error(f"{option} должен быть больше нуля")


def exit_code_for_results(results: Sequence[ModelResult]) -> int:
    if fastest_result(results):
        return 0
    infrastructure_errors = {
        "AUTH_ERROR",
        "CONNECTION_ERROR",
        "REQUEST_ERROR",
        "SERVER_ERROR",
    }
    return 2 if results and all(result.status in infrastructure_errors for result in results) else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _positive(parser, args.workers, "--workers")
    _positive(parser, args.timeout, "--timeout")
    _positive(parser, args.runs, "--runs")
    _positive(parser, args.max_tokens, "--max-tokens")

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Ошибка: OPENROUTER_API_KEY не задан.", file=sys.stderr)
        return 2

    session = requests.Session()
    headers = build_headers(api_key)
    try:
        print("Получаем список бесплатных моделей...")
        models = get_free_models(session, headers, args.timeout)
        if args.model:
            filters = [value.lower() for value in args.model]
            models = [
                model for model in models if any(value in model.id.lower() for value in filters)
            ]
        if not models:
            print("Подходящие бесплатные модели не найдены.")
            return 1

        print(f"Найдено моделей: {len(models)}. Запускаем проверку...\n")
        results = sort_results(
            benchmark_models(
                session,
                headers,
                models,
                args.workers,
                args.runs,
                args.prompt,
                args.max_tokens,
                args.timeout,
            ),
            args.sort,
        )
        print_results(results, args.runs)
        winner = fastest_result(results)
        if winner:
            print(
                f"\nСамая быстрая: {winner.model.id} "
                f"({winner.median_latency:.3f} с, {winner.successful_runs}/{args.runs} успешно)"
            )
        else:
            print("\nНет моделей с успешным ответом.")

        if args.json:
            export_results(args.json, results)
            print(f"Результаты сохранены: {args.json}")
        return exit_code_for_results(results)
    except ScannerError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"Ошибка записи результатов: {exc}", file=sys.stderr)
        return 2
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
