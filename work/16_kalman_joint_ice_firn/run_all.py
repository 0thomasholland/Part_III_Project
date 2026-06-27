from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "joint_kalman.py",
    "knockout_test.py",
    "bore_sensitivity.py",
    "multi_seed_knockout.py",
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
    max_workers = min(int(os.environ.get("WORK16_MAX_WORKERS", "2")), len(SCRIPTS))
    failures: list[tuple[str, int]] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_script, script_name): script_name
            for script_name in SCRIPTS
        }
        for future in as_completed(futures):
            script_name, returncode, output = future.result()
            print(f"\n=== {script_name} ===")
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


if __name__ == "__main__":
    main()
