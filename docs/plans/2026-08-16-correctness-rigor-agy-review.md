# AGY Review: Correctness and Rigor Plan

**Reviewed commit:** `4226a33c8184ac26b620e7bcb9d6d8da6123c2f7`

**Reviewer:** Gemini via AGY

**Verdict:** `LGTM WITH NON-BLOCKING NOTES`

The review ran read-only in a clean detached worktree. Its reported workspace
and commit matched the requested target, and the worktree remained at the same
clean commit afterward.

## Verified findings and resolutions

1. The proposed MH boundary test checked only that a particle remained inside
   support, so an incorrect reflection implementation could pass. The plan now
   uses a deterministic crossing proposal and requires the particle to remain
   exactly unchanged with zero acceptances.
2. The tempering task described resampling only before `beta=1`. A low-ESS
   terminal stage could therefore leave the next update below its ESS target
   with no valid bisection root. The plan now resamples and rejuvenates whenever
   ESS reaches the threshold, including at `beta=1`.
3. Rejection sampling and MH can normally generate zero-distance candidates.
   Raw potential evaluation could emit divide warnings before rejecting them.
   The plan now requires local NumPy error-state handling and an explicit
   invalid mask, while public direct calls remain strict errors.
4. The retuned density truth did not state the matching amplitude prior. The
   plan now fixes the development prior to `U(0.001, 0.030)` and conditions it
   on the same weak-field support.
5. The API support test now checks support after multiple updates as well as at
   initialization.
6. Packaged demo modules now explicitly select Matplotlib's headless `Agg`
   backend before importing `pyplot`.

No reviewer claim was accepted without checking it against the design, plan,
and current repository behavior.
