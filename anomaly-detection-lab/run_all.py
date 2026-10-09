"""Run every technique script in order and print a comparison table."""

import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = sorted(f for f in os.listdir(ROOT)
                 if f[:2].isdigit() and f.endswith(".py"))


def main():
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    failed = []
    for script in SCRIPTS:
        print(f"\n{'=' * 70}\n>>> {script}\n{'=' * 70}")
        t0 = time.time()
        r = subprocess.run([sys.executable, os.path.join(ROOT, script)],
                           cwd=ROOT, env=env)
        print(f"<<< {script} done in {time.time() - t0:.1f}s "
              f"(exit {r.returncode})")
        if r.returncode != 0:
            failed.append(script)

    results_path = os.path.join(ROOT, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        print(f"\n{'=' * 70}\nSUMMARY (threshold chosen at best raw F1)\n{'=' * 70}")
        hdr = f"{'method':<32}{'P':>7}{'R':>7}{'F1':>7}{'PA-F1':>8}"
        print(hdr + "\n" + "-" * len(hdr))
        for name in sorted(results):
            m = results[name]
            print(f"{name:<32}{m['precision']:>7.3f}{m['recall']:>7.3f}"
                  f"{m['f1']:>7.3f}{m['pa_f1']:>8.3f}")
    if failed:
        print("\nFAILED:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
