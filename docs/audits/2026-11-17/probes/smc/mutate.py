"""Apply sampler mutants one at a time in the audit worktree and run the test group.

Run from the worktree root: uv run python <this file>
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(".worktrees/audit-smc")
TESTS = [
    "tests/test_smc_rigorous.py",
    "tests/test_smc_api.py",
    "tests/test_inference.py",
    "tests/test_weight_intervals.py",
    "tests/test_weight_normalization.py",
    "tests/test_weight_filter.py",
    "tests/test_weight_tools.py",
    "tests/test_api.py",
]
INF = "src/clocks/inference.py"

MUTANTS = {
    "M1_mh_targets_beta1": (
        INF,
        "        current = np.array(particles, dtype=np.float64, copy=True)\n",
        "        beta = 1.0\n        current = np.array(particles, dtype=np.float64, copy=True)\n",
    ),
    "M2_mh_targets_beta0": (
        INF,
        "        current = np.array(particles, dtype=np.float64, copy=True)\n",
        "        beta = 0.0\n        current = np.array(particles, dtype=np.float64, copy=True)\n",
    ),
    "M3_accept_all_in_support": (
        INF,
        "                accept = np.log(uniforms) < np.minimum(0.0, log_alpha)\n",
        "                accept = np.isfinite(proposal_target) & (uniforms >= 0)\n",
    ),
    "M6_no_resampling_drop_weights": (
        INF,
        "                indices = self._resample_indices(weights)\n",
        "                indices = np.arange(self.n_particles)\n",
    ),
    "M9_mh_drops_prior": (
        INF,
        "        result[valid] = prior[valid] + target_stats.log_likelihood(",
        "        result[valid] = 0.0 * prior[valid] + target_stats.log_likelihood(",
    ),
    "M21_mh_ratio_sign_flipped": (
        INF,
        "            log_alpha = proposal_target - current_target\n",
        "            log_alpha = current_target - proposal_target\n",
    ),
    "M22_stale_loglik_after_move": (
        INF,
        "            observation_ll = self._observation_log_likelihood(particles, observation)\n",
        "            observation_ll = (\n                self._observation_log_likelihood(particles, observation)\n                if beta == 0.0\n                else observation_ll\n            )\n",
    ),
    "M24_target_not_updated_on_accept": (
        INF,
        "            current_target[accept] = proposal_target[accept]\n",
        "",
    ),
    "M25_beta_fraction_ignored_in_mh": (
        INF,
        "        target_stats = completed.with_fraction(observation.rates, beta)\n",
        "        target_stats = completed.with_fraction(observation.rates, 1.0 if beta > 0 else 0.0)\n",
    ),
    "M26_next_beta_returns_high": (
        INF,
        '    if low <= beta:\n        raise RuntimeError("adaptive tempering could not make positive progress")\n    return low\n',
        '    if low <= beta:\n        raise RuntimeError("adaptive tempering could not make positive progress")\n    return min(1.0, beta + 2.0 * (high - beta))\n',
    ),
    "M27_evidence_ignores_base_weights": (
        INF,
        "            increment = log_increment + evidence_offset\n",
        "            increment = log_increment + evidence_offset - float(np.log(np.sum(weights**2)) * 0)\n            increment = float(logsumexp(delta * observation_ll[np.isfinite(observation_ll)]) - np.log(np.isfinite(observation_ll).sum()))\n",
    ),
}


def clean() -> None:
    for cache in ROOT.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


def porcelain() -> str:
    return subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    ).stdout


def run_tests() -> str:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run(
        ["uv", "run", "pytest", "-q", "-p", "no:cacheprovider", "-rf", *TESTS],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    ).stdout
    failed = re.findall(r"^FAILED (\S+)", out, re.M)
    summary = out.strip().splitlines()[-1] if out.strip() else "<no output>"
    return summary + "\n" + "\n".join("    " + f for f in failed)


def main() -> None:
    selected = sys.argv[1:] or list(MUTANTS)
    for name in selected:
        path, old, new = MUTANTS[name]
        target = ROOT / path
        assert porcelain() == "", f"tree not pristine before {name}: {porcelain()}"
        original = target.read_text()
        assert original.count(old) == 1, f"{name}: anchor count {original.count(old)}"
        try:
            target.write_text(original.replace(old, new))
            clean()
            print(f"== {name}\n{run_tests()}", flush=True)
        finally:
            target.write_text(original)
            clean()
        assert porcelain() == "", f"tree not restored after {name}"


if __name__ == "__main__":
    main()
