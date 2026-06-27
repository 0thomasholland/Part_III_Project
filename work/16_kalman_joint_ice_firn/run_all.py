from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from parallel_utils import run_jobs_in_threads


ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    (
        "joint_kalman.py",
        "Full joint Kalman filter and smoother baseline.",
    ),
    (
        "knockout_test.py",
        "Single-seed sensor knockout comparison.",
    ),
    (
        "bore_sensitivity.py",
        "Bore count and revisit sensitivity sweep.",
    ),
    (
        "multi_seed_knockout.py",
        "Multi-seed knockout robustness sweep.",
    ),
]


def _run_script(script_name: str) -> tuple[str, int, str]:
    completed = subprocess.run(
        [sys.executable, str(ROOT / script_name)],
        cwd=ROOT.parent.parent,
        capture_output=True,
        text=True,
    )
    output = completed.stdout
    if completed.stderr:
        output = f"{output}\n{completed.stderr}".strip()
    return script_name, completed.returncode, output


def main() -> None:
    failures: list[tuple[str, int]] = []
    descriptions = dict(SCRIPTS)

    print("Running work16 experiment suite:")
    for script_name, description in SCRIPTS:
        print(f"- {script_name}: {description}")

    for script_name, returncode, output in run_jobs_in_threads(
        [script_name for script_name, _ in SCRIPTS],
        _run_script,
    ):
        print(f"\n=== {script_name} ===")
        print(descriptions[script_name])
        if output:
            print(output)
        print(f"exit_code={returncode}")
        if returncode != 0:
            failures.append((script_name, returncode))

    if failures:
        raise SystemExit(
            "Failed scripts: "
            + ", ".join(
                f"{script_name} ({returncode})"
                for script_name, returncode in failures
            )
        )

    print("All work16 scripts completed successfully.")


if __name__ == "__main__":
    main()
