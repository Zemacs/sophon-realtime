"""Release gate (Round 4k / R-9). Run from live/:  python tests/release_gate.py

Fails (exit 1) unless ALL of these hold (exit 2 = a script problem such as a harness timeout, which is not a verdict on the code):
  * ``node`` is installed (the page behaviour tests need it; without it they are SKIPPED, and a skipped page test must never count as a pass),
  * the page behaviour tests ran and none was skipped,
  * the whole suite passes with every ``SOPHON_LIVE_*`` variable removed from the environment,
  * the mutation script exits 0 (every mutant killed),
  * the blind-runner mutation script (tests/mutation_check_blind_runner.py) exits 0 (every runner mutant killed).
There is no CI in this repo: run this wherever a release or deploy is decided (deplux release check / deploy script)."""
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARNESS_TIMEOUT_MARKER = "the page harness did not finish in"      # same text as tests/test_r4j_page_behaviour.py
PYTEST = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]


def clean_env() -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SOPHON_LIVE_")}
    env["PYTHONPATH"] = "."
    return env


def fail(msg: str) -> None:
    print("RELEASE GATE FAILED:", msg)
    sys.exit(1)


def main() -> None:
    if not shutil.which("node"):
        fail("node is not installed: the page behaviour tests cannot run (tests/test_r4j_page_behaviour.py)")
    r = subprocess.run(PYTEST + ["-rs", "tests/test_r4j_page_behaviour.py"], cwd=ROOT, capture_output=True, text=True, env=clean_env())
    out = r.stdout + r.stderr
    if HARNESS_TIMEOUT_MARKER in out:
        print("RELEASE GATE SCRIPT ERROR: the page harness timed out (machine busy?); this says nothing about the page. Re-run on an idle machine.\n" + out[-800:])
        sys.exit(2)
    if r.returncode != 0 or re.search(r"\d+ skipped", out) or not re.search(r"\d+ passed", out):
        fail("the page behaviour tests failed or were skipped:\n" + out[-1500:])
    r = subprocess.run(PYTEST + ["tests"], cwd=ROOT, capture_output=True, text=True, env=clean_env())
    print((r.stdout.strip().splitlines() or [""])[-1])
    if r.returncode != 0:
        fail("the full suite failed with SOPHON_LIVE_* cleared:\n" + (r.stdout + r.stderr)[-2000:])
    r = subprocess.run([sys.executable, "tests/mutation_check_r4c.py"], cwd=ROOT, capture_output=True, text=True, env=clean_env())
    print((r.stdout.strip().splitlines() or [""])[-1])
    if r.returncode != 0:
        fail("the mutation script did not exit 0 (survivor or script error):\n" + r.stdout[-2000:])
    r = subprocess.run([sys.executable, "tests/mutation_check_blind_runner.py"], cwd=ROOT, capture_output=True, text=True, env=clean_env())
    print((r.stdout.strip().splitlines() or [""])[-1])
    if r.returncode != 0:
        fail("the blind-runner mutation script did not exit 0 (survivor or script error):\n" + r.stdout[-2000:])
    print("RELEASE GATE OK")


if __name__ == "__main__":
    main()
