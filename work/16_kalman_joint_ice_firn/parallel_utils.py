from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import TypeVar


T = TypeVar("T")
R = TypeVar("R")


def resolve_max_workers(total_jobs: int) -> int:
    return min(int(os.environ.get("WORK16_MAX_WORKERS", "2")), total_jobs)


def run_cases_in_pool(
    cases: Iterable[T],
    evaluate_case: Callable[[T], R],
) -> list[R]:
    case_list = list(cases)
    if not case_list:
        return []

    results: list[R] = []
    with ProcessPoolExecutor(
        max_workers=resolve_max_workers(len(case_list))
    ) as executor:
        futures = [executor.submit(evaluate_case, case) for case in case_list]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def run_jobs_in_threads(
    jobs: Iterable[T],
    run_job: Callable[[T], R],
) -> list[R]:
    job_list = list(jobs)
    if not job_list:
        return []

    results: list[R] = []
    with ThreadPoolExecutor(
        max_workers=resolve_max_workers(len(job_list))
    ) as executor:
        futures = [executor.submit(run_job, job) for job in job_list]
        for future in as_completed(futures):
            results.append(future.result())
    return results
