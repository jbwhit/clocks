# Revision-3 independent review and evidence

This is a design/prototype checkpoint, not production or merge approval.
Jonathan explicitly chose correct exact-subnormal rounding over preserving
all formerly float-normal outputs. The [design](../2026-09-14-normalizer-r3-design.md)
implements that choice with an explicit raise-on-budget-exhaustion contract.

## Reviewed snapshot and environment

Review base: `ff5d39aa4f4c69dfd752e3a3835cea6dd750ca39`.
The proposed files were copied to the isolated detached worktree
`.worktrees/r3-review` within the normalizer worktree. They remained untracked
during read-only review, with contents pinned by SHA-256:

| File | SHA-256 |
|---|---|
| `prototype.py` | `3898927019a6fffe0ececdbbcf875cdf02da5425bac4ccad75bc59a695e27865` |
| `dd-analysis.md` | `7d4856ce02af05b6e139e9001e066615235ffdbbedd1fd8cfd7e0b91c23f1247` |
| Design, before final evidence/status edits | `c34dddf58ccbac86e819a727dda0c18586a788c7b72064dedc1d51da85049179` |

The controller recorded HEAD, `git status --short`, and these hashes before and
after AGY; all matched. AGY reported the expected cwd, HEAD, status, and target
line `R3 full-exact-subnormal coverage; both-midpoint decision; trim=2^-350`.
No review environment was installed, copied, activated, or synchronized.
Execution checks used `uv run --no-sync` in the original normalizer worktree;
the harness asserts `clocks.inference.__file__` resolves within that checkout.

## Gemini via AGY

Reviewer: `gemini-3.8-flash-high`, high effort, conversation
`bbbbdfa1-bf4c-49ea-8212-70e668a6fb23`. The actual binary was
`/Users/jonathan/.local/bin/agy`. No dangerous-permission bypass or new allow-rule
was used. The initial attempt encountered a denied `find docs/plans -type f`;
after inspecting the CLI log, the retry identified the exact denied command and
continued with existing permitted read commands.

The [original report](agy-review.md.txt) returned **SOUND ENOUGH TO IMPLEMENT**.
It checked candidate coverage, DD operation order and composed bounds, both
rounding boundaries, Decimal enclosure, cap policy, and cost qualifications.
The controller challenged these claims rather than accepting the report whole:

1. **Slicing allocation:** NumPy basic-strided slices share source memory.
   Direct `np.shares_memory` checks confirmed both even and odd views. Arithmetic
   still allocates, but slicing alone does not justify Cython or preallocation.
   Gemini retracted this performance suggestion.
2. **Normal-weight identity and ESS:** the revised contract allows near-normal
   corrections. The design does not establish an accuracy bound on NumPy exp,
   or a theorem that all ESS results or filter trajectories remain identical.
   The narrower observation is that the tiny corrected results square to zero.
   Gemini retracted its bitwise-normal-identity claim. Its follow-up reference
   to a standard 0.5–1 ulp libm error is not an established premise here and is
   not used by the design or certificate.
3. **Helper separation:** accepted. Production integration must use distinct
   fast/strict helper bindings; the prototype's function-global injection is
   only an experimental harness.
4. **Hash attribution:** SHA-256 verification was performed by the controller,
   not AGY. Gemini acknowledged the correction.

The [follow-up](agy-clarification.md.txt) reconfirmed **SOUND ENOUGH TO IMPLEMENT**.
The report's strong proof language remains a human-readable, reasoned review;
it is not machine-checked verification.

## Independent Codex numerical review

A separate reviewer, who did not author the revised arithmetic analysis,
reviewed the same frozen prototype and the test-control follow-up. Its verdict:

> SOUND ENOUGH TO IMPLEMENT
>
> No blocking numerical findings in prototype hash
> `3898927019a6fffe0ececdbbcf875cdf02da5425bac4ccad75bc59a695e27865`.
>
> Candidate selection covers every exact subnormal weight independently of
> NumPy’s exponential accuracy. Trimming, normalization, the lattice argument,
> and scaling match the implemented operations. The composed error bounds leave
> sufficient room below the actual decision radius. Both rounding boundaries
> use conservative distance bounds; ambiguous cases escalate. Decimal
> exponential enclosures, directed accumulation/division, omitted-tail allowance,
> and integer endpoint rounding preserve enclosure.
>
> The ESS discussion appropriately limits its conclusion. Accepted stages can
> still exhaust the precision budget; global trajectory invariance is not claimed.
>
> Independent measurements: exact rational checks passed for 1,008 certified
> rounding cells and 320 division edge cases. Sixteen boundary cases remained
> ambiguous. These are supplemental checks, not the certificate’s basis.
>
> This verdict concerns readiness to implement under the stated arithmetic
> assumptions and explicit raise-on-exhaustion contract. It does not approve
> production code, merging, or deployment.

## Test-control correction and evidence scope

The independent reviewer also checked the CI step, injected-suite harness,
mutation propagation, and corrected negative control in the final working
tree and reported no blocking harness or CI findings.

An initial combined injected-suite run exposed a harness defect: the intended
unsplit negative control inherited the already-split `_next_beta` binding, so
its expected cap failure did not occur. The test now constructs its unsplit
function with the module globals explicitly, before installing the split.
Both reviewers considered the follow-up; prototype and proof hashes did not
change. Earlier aggregate mutation failure counts that included this unrelated
control failure are not the final evidence.

Injection is confined to the pytest interpreter. Tests spawning fresh CLI
processes continue to exercise the branch source; a green injected run is not
a claim that the shipped binary contains the prototype. Production integration
must run the full suite again against actual implementation files.

## Final verification

Local baseline: Ruff formatter check and lint passed; package plus 31 design
tests returned **929 passed, 4 xfailed, 2 slow deselected**. The separate slow
gate returned **2 passed**. The four xfails belong to the unchanged branch
production implementation, not the prototype.

Each in-memory mutation ran all 935 package and design tests, including slow
replays, with the four known xfails converted into ordinary assertions.
The controller inspected the failures, not just exit status:

| Mutation | Result | Intended failing evidence |
|---|---|---|
| Old midpoint decision | 1 failed, 934 passed | Round-2 lower-midpoint result is one quantum wrong |
| Preserve float-normal classification | 1 failed, 934 passed | Exact-subnormal seam output is wrong |
| Remove DD radius | 4 failed, 931 passed | Wrong zero/minimum-quantum decision, a crossing interval accepted, and two real cap controls stop raising |
| Collapse Decimal exp intervals | 4 failed, 931 passed | Independent high-precision values fall outside four purported enclosures |

All four runs completed with pytest exit 1. Full outputs are archived in
[mutation-results.txt](mutation-results.txt). These tests demonstrate sensitivity
to the selected defects; they do not prove the mathematical bounds.

The final unmodified injected full suite, including both slow acceptance tests,
returned **935 passed in 287.61 seconds**, exit 0. Formatter and lint checks were
rerun after recording this evidence. No production source changed during the
review or verification; final documentation records the evidence and narrows
the scope of claims, without changing the reviewed arithmetic.
