# Fedot.Industrial: backlog issues for the forecasting track

## Scope

This backlog is derived from the uploaded planning note, the current code archive, and the supporting theory around
mSSA, HAVOK, OKHS/fDMD, and reduced-order decomposition.

Primary scope:

- time-series forecasting;
- near-term productizable work inside `Fedot.Industrial`;
- issues that are still open **after** reconciling the roadmap with the uploaded code archive.

## What is already present in the archive

The codebase already contains a nontrivial forecasting substrate:

- classical forecasting modules, including `ssa_forecaster.py`, lagged/eigen/topological forecasters;
- DMD and DMD-forecasting layers;
- the OKHS/fDMD stack with shared policies in `okhs_common.py`;
- `okhs_forecasting.py` and `okhs_forecasting_torch.py`;
- deep-OKHS components under `method_impl/deep_okhs/`;
- decomposition utilities such as SVD and column sampling;
- benchmark scaffolding under `benchmark/v2/`.

Important consequence: several older roadmap items are now obsolete or partially completed. In particular, the current
archive already has typed OKHS policies/enums, explicit `q_policy`, projected-latent OKHS vocabulary, and a benchmark
layer. The backlog below therefore avoids reopening work that appears already implemented.

## Architectural reading

The planning note makes the intended forecasting stack explicit:

```text
raw TS
-> regime diagnostics
-> representation block
-> latent forecast block
-> residual correction
-> rolling recalibration
```

and recommends three immediate products for the forecasting direction:

1. `mSSA`
2. `HAVOK`
3. a unified `OKHS/fDMD` layer

That matches the external theory well:

- mSSA gives a stacked multivariate Page/Hankel representation with finite-sample forecasting guarantees under a
  spatio-temporal factor model;
- HAVOK gives a delay-coordinate linear model with an intermittent forcing channel for regime switches / bursts;
- the current OKHS/fDMD line is the natural higher-capacity continuation when fixed coordinates are insufficient.

## Prioritization logic

- **P0**: removes ambiguity or dead code in the current forecasting path.
- **P1**: adds missing core forecasting capabilities.
- **P2**: improves robustness, diagnostics, or generalization.
- **R&D**: worth starting only after the forecasting core is stable.

---

# Issue 1. Resolve the status of `ssa_forecaster.py`

**Priority:** P0  
**Type:** architecture / maintenance  
**Labels:** `forecasting`, `tech-debt`, `api`, `ssa`

## Problem

The current archive still exposes a legacy SSA forecaster, but the implementation status is ambiguous: the module
contains substantial prediction logic while the main `fit(...)` entry point is effectively unfinished. That creates API
uncertainty and makes benchmarking unfair, because it is not obvious whether SSA is a supported baseline, a deprecated
path, or a partially migrated component.

## Why now

Before adding mSSA or HAVOK, the baseline layer must be made unambiguous. Otherwise the forecasting registry contains
overlapping or half-supported modes.

## Scope

Choose one of the following and implement it fully:

1. **finish** the legacy SSA forecaster and keep it public;
2. **deprecate** it in favor of a new shared SSA/mSSA backend;
3. **wrap** it as a compatibility adapter over the new backend.

## Affected modules

- `fedot_ind/core/models/ts_forecasting/ssa_forecaster.py`
- forecasting registry / model repository bindings
- docs and examples for forecasting

## Acceptance criteria

- there is exactly one documented status for legacy SSA: supported, deprecated, or compatibility-only;
- `fit/predict` behavior is covered by unit tests;
- forecasting examples no longer rely on undocumented SSA behavior;
- benchmark tables do not silently include a broken SSA variant.

## Dependencies

None.

---

# Issue 2. Create a shared trajectory embedding backend

**Priority:** P0  
**Type:** architecture  
**Labels:** `forecasting`, `representation`, `refactor`

## Problem

SSA, mSSA, HAVOK, DMD, and OKHS all depend on the same family of trajectory constructions:

- Hankel or Page embedding;
- window policy;
- stride policy;
- diagonal averaging or decoding;
- optional low-rank truncation.

Right now these ideas are scattered across separate modules. That increases duplicated logic and makes new forecasters
expensive to add.

## Scope

Introduce a shared backend for trajectory representations with a minimal but explicit API:

- `build_hankel(...)`
- `build_page(...)`
- `stack_multivariate(...)`
- `decode_diagonal_average(...)`
- `decode_page(...)`
- `estimate_window(...)`
- optional `truncate_rank(...)`

## Affected modules

- `fedot_ind/core/models/ts_forecasting/*`
- `fedot_ind/core/models/kernel/okhs_common.py`
- `fedot_ind/core/operation/transformation/data/*`

## Acceptance criteria

- SSA, mSSA, and HAVOK can call the same embedding primitives;
- window/stride choices are logged in diagnostics;
- no forecaster reimplements Page/Hankel construction ad hoc.

## Dependencies

Issue 1.

---

# Issue 3. Implement `mssa_forecaster.py`

**Priority:** P1  
**Type:** feature  
**Labels:** `forecasting`, `multivariate`, `mssa`

## Problem

The current planning note explicitly prioritizes mSSA for multivariate forecasting, but the uploaded archive does not
contain a dedicated mSSA forecaster. That is a real gap: the framework already supports multivariate industrial
problems, but the low-rank stacked embedding baseline is missing.

## Theory anchor

The uploaded mSSA paper defines the core construction clearly: each channel is converted into an `L x (T/L)` Page
matrix, then the matrices are stacked column-wise into a single `L x N(T/L)` matrix, followed by hard singular value
thresholding and a linear forecast stage.

## Scope

Implement a first production baseline with:

- stacked Page matrix construction;
- missing-value friendly normalization;
- HSVT-based denoising;
- vector forecast head on the denoised representation;
- rolling-origin evaluation support.

## Design constraints

- keep v1 simple: Page-matrix variant first, not full signal-extraction SSA;
- support both `channel_independent=False` and explicit coupling;
- expose `L`, rank, threshold, and horizon in diagnostics.

## Affected modules

- new: `fedot_ind/core/models/ts_forecasting/mssa_forecaster.py`
- shared embedding backend from Issue 2
- examples / docs / benchmark configs

## Acceptance criteria

- works on multivariate arrays with `N > 1`;
- returns stable forecasts on noisy and partially missing multivariate series;
- benchmark entry exists;
- comparison against univariate SSA and VAR-style baselines is reproducible.

## Dependencies

Issue 2.

---

# Issue 4. Add `regime_diagnostics.py`

**Priority:** P1  
**Type:** feature / diagnostics  
**Labels:** `forecasting`, `diagnostics`, `regime-detection`

## Problem

The planning note proposes regime-aware forecasting, but there is no explicit lightweight diagnostics block that
estimates whether a series is:

- strongly periodic;
- broadband / weakly predictable;
- regime-switching;
- locally linearizable.

Without that layer, model routing remains manual.

## Scope

Add a small diagnostics module returning structured metadata:

- ACF decay rate;
- dominant period / spectral concentration;
- entropy proxy or spectral flatness;
- instability proxy from local DMD eigenvalue drift;
- optional change-point / switching score.

## Affected modules

- new: `fedot_ind/core/models/ts_forecasting/regime_diagnostics.py`
- forecasting orchestration layer
- benchmark artifact writer

## Acceptance criteria

- diagnostics can run before model selection;
- outputs are serializable into benchmark JSON;
- docs explain how the diagnostics affect model routing.

## Dependencies

Issue 2.

---

# Issue 5. Implement `havok_forecaster.py`

**Priority:** P1  
**Type:** feature  
**Labels:** `forecasting`, `koopman`, `havok`, `dmd`

## Problem

HAVOK is explicitly recommended in the planning note for regime-switching and rare-event series, but there is no
dedicated forecaster in the archive. This is the most important missing model between classical SSA and OKHS/fDMD.

## Theory anchor

HAVOK splits delay coordinates into:

- a nearly linear subsystem on the first `r-1` coordinates;
- a final forcing coordinate `v_r` capturing intermittent nonlinear events.

That gives a practical predictor for systems that are locally linear between switching events.

## Scope

Implement a first forecaster with:

- Hankel embedding;
- SVD to eigen-delay coordinates;
- derivative estimation in latent coordinates;
- regression for `(A, B)` in `dv/dt = Av + B v_r`;
- short-horizon forecast with forcing-aware correction.

## Design constraints

- ship as a forecast model, not only as an analysis notebook;
- expose forcing activity as a first-class output in diagnostics;
- keep the latent rank / window choice explicit and inspectable.

## Affected modules

- new: `fedot_ind/core/models/ts_forecasting/havok_forecaster.py`
- DMD utilities
- benchmark layer

## Acceptance criteria

- can forecast at least univariate series via delay embedding;
- emits forcing diagnostics;
- benchmark artifact includes active/inactive forcing intervals;
- evaluated on at least one synthetic switching dataset and one real dataset.

## Dependencies

Issues 2 and 4.

---

# Issue 6. Add HAVOK event-aware evaluation artifacts

**Priority:** P1  
**Type:** feature / benchmark  
**Labels:** `benchmark`, `forecasting`, `havok`, `diagnostics`

## Problem

If HAVOK is benchmarked only by aggregate RMSE/MAE, the main advantage of the method is invisible. Its real value is
often in event precursors and regime-switch diagnostics.

## Scope

Extend the benchmark layer with artifacts such as:

- forcing activity timeline;
- boundary overlay near switching points;
- precision/recall for event precursor detection when labels exist;
- latent coordinate plots for debugging.

## Affected modules

- `benchmark/v2/forecasting.py`
- `benchmark/v2/analytics.py`
- new visualization helpers if needed

## Acceptance criteria

- HAVOK runs produce event-aware artifacts by default;
- metrics distinguish forecast quality inside calm vs forcing-active intervals;
- publication-ready plots are saved as named benchmark assets.

## Dependencies

Issue 5.

---

# Issue 7. Add a regime-aware routing policy across forecasting models

**Priority:** P1  
**Type:** feature / orchestration  
**Labels:** `forecasting`, `automl`, `routing`

## Problem

The current roadmap recommends a regime-aware pipeline, but model choice still appears largely manual. That prevents
systematic comparison between:

- AR or lagged baselines;
- SSA / mSSA;
- DMD / HAVOK;
- OKHS/fDMD.

## Scope

Implement a small routing policy that proposes candidate models from diagnostics:

- strong periodic + low noise -> SSA / mSSA
- switching / bursts -> HAVOK
- smooth locally linear latent structure -> DMD / OKHS
- weak structure -> lagged AR-style fallback

This can start as a heuristic recommender before any full AutoML integration.

## Affected modules

- forecasting meta-layer
- `api` entry points if exposed to users
- benchmark configs

## Acceptance criteria

- routing policy is deterministic and documented;
- every recommendation can be explained from diagnostics;
- fallback path exists when diagnostics are inconclusive.

## Dependencies

Issues 3, 4, 5.

---

# Issue 8. Fix the over-smoothing failure mode in projected OKHS/fDMD

**Priority:** P1  
**Type:** bug / model-quality  
**Labels:** `forecasting`, `okhs`, `fdmd`, `quality`

## Problem

The internal refactor recap indicates that the major unresolved problem is still forecast degeneration on part of the
M4-style series: the model may produce an overly smooth decaying trajectory instead of preserving local oscillatory
structure.

## Why this matters

This is the strongest empirical objection to deploying projected OKHS/fDMD as a default forecasting block.

## Scope

Turn the problem into a dedicated issue with a reproducible suite:

- identify failure cohorts;
- compare `reconstructed` vs `projected` trajectory representation;
- study window/rank/q interactions;
- test residual correction and boundary alignment;
- add explicit anti-smoothing diagnostics.

## Affected modules

- `okhs_common.py`
- `okhs_forecasting.py`
- `okhs_forecasting_torch.py`
- benchmark datasets/configs

## Acceptance criteria

- at least one reproducible regression test captures the smoothing failure;
- proposed fix measurably reduces the failure rate on the selected cohort;
- diagnostics expose when the model is collapsing to a monotone envelope.

## Dependencies

Existing OKHS stack.

---

# Issue 9. Integrate Deep OKHS decoupled spectral training into the forecasting API

**Priority:** P2  
**Type:** feature / R&D bridge  
**Labels:** `forecasting`, `okhs`, `deep-learning`, `r-and-d`

## Problem

The archive already contains deep-OKHS building blocks, but they are not yet presented as a coherent forecasting path:

1. learn a nonlinear latent space with weak/integral loss;
2. freeze the encoder;
3. run analytical fractional DMD in the learned space.

This is precisely the main research continuation in the uploaded theorem and mathematical-engine notes.

## Scope

Create an experimental but callable pipeline:

- `method="deep_projected_fdmd"` or equivalent;
- training phase with encoder/decoder + `W`;
- freeze / export latent trajectories;
- inference phase via analytical fDMD on frozen latent space.

## Design constraints

- do **not** backpropagate through the spectral stage in v1;
- log the separation between training-time surrogate `W` and inference-time spectral operator;
- keep this path explicitly marked experimental.

## Affected modules

- `method_impl/deep_okhs/*`
- `okhs_forecasting_torch.py`
- API docs and examples

## Acceptance criteria

- one entry point can run the full two-phase pipeline;
- artifacts clearly separate phase-1 and phase-2 outputs;
- benchmark compares against projected analytical OKHS on the same datasets.

## Dependencies

Issue 8 first, otherwise the experimental branch will be hard to interpret.

---

# Issue 10. Extend randomized decomposition utilities

**Priority:** P2  
**Type:** feature  
**Labels:** `decomposition`, `sampling`, `rsvd`, `cur`

## Problem

The planning note correctly identifies randomized decomposition as a reusable substrate for both forecasting and future
large-tensor work. The archive has decomposition utilities, but not yet a full structure-aware sampling layer.

## Scope

Extend current decomposition code with:

- randomized SVD;
- block power or Krylov iterations;
- leverage-score column sampling;
- block / mode-wise sampling;
- reconstruction diagnostics.

## Affected modules

- `method_impl/svd_decompostion.py`
- `method_impl/column_sampling_decomposition.py`
- new helper modules if needed

## Acceptance criteria

- decomposition methods expose reconstruction error diagnostics;
- interfaces are consistent across exact and randomized variants;
- at least one forecasting pipeline can optionally use rSVD for large embeddings.

## Dependencies

None.

---

# Issue 11. Add `gappy_pod_reconstruction.py` and `deim_selector.py`

**Priority:** P2  
**Type:** feature  
**Labels:** `rom`, `gappy-pod`, `deim`, `selective-computation`

## Problem

The roadmap proposes gappy POD / DEIM as selective-computation blocks. These are not core to v1 forecasting, but they
are highly reusable for:

- missing-data reconstruction;
- partial-channel time-series inference;
- future tensor / cache compression paths.

## Scope

Implement:

- gappy POD reconstruction;
- DEIM interpolation point selection;
- explicit condition-number diagnostics for selected measurements.

## Affected modules

- new: `gappy_pod_reconstruction.py`
- new: `deim_selector.py`
- optional: `rom_feature_selector.py`

## Acceptance criteria

- can reconstruct from sparse observations in a low-rank basis;
- reports conditioning / reconstruction stability;
- benchmark notebook or script demonstrates the operator.

## Dependencies

Issue 10 is helpful but not mandatory.

---

# Issue 12. Expand forecasting test coverage

**Priority:** P0  
**Type:** testing  
**Labels:** `tests`, `forecasting`, `quality`

## Problem

The archive has tests, but dedicated coverage for OKHS/fDMD, mSSA, and HAVOK is still too thin relative to the
complexity of the stack.

## Scope

Add layered tests:

### Unit

- shape contracts for embeddings;
- deterministic behavior of window/rank/q policies;
- Gram / operator invariants where applicable.

### Integration

- honest rolling-origin forecast tests;
- synthetic regime-switching cases for HAVOK;
- multivariate tests for mSSA;
- forecast quality sanity checks against simple baselines.

### Regression

- specific M4-like failure cohorts for the OKHS smoothing issue.

## Affected modules

- `tests/unit/core/models/*`
- `tests/integration/ts_forecast/*`

## Acceptance criteria

- every new forecasting model ships with both unit and integration tests;
- examples used in docs can be executed in CI or nightly runs;
- at least one regression suite guards against old forecast collapse modes.

## Dependencies

Issues 3, 5, 8.

---

## Recommended delivery order

### Wave 1

1. Issue 1 — SSA status
2. Issue 2 — shared embedding backend
3. Issue 12 — minimum forecasting test scaffold

### Wave 2

4. Issue 3 — mSSA
5. Issue 4 — regime diagnostics
6. Issue 5 — HAVOK
7. Issue 6 — HAVOK benchmark artifacts

### Wave 3

8. Issue 8 — projected OKHS smoothing failure
9. Issue 7 — regime-aware routing
10. Issue 9 — Deep OKHS two-phase integration

### Wave 4

11. Issue 10 — randomized decomposition extensions
12. Issue 11 — gappy POD / DEIM

---

## Suggested milestone structure

### Milestone A — Forecasting core stabilization

- Issues 1, 2, 8, 12

### Milestone B — New forecasting models

- Issues 3, 4, 5, 6, 7

### Milestone C — Research extension

- Issue 9

### Milestone D — Selective computation / compression

- Issues 10, 11

---

## Ready-to-use issue template fields

For every issue above, I recommend the same issue form fields:

- **Problem**
- **Why now**
- **Scope**
- **Affected modules**
- **Acceptance criteria**
- **Dependencies**
- **Out of scope**
- **Benchmark / datasets**
- **Risks**

This is compatible with GitHub Issues best practice: issues as trackable work items, issue templates for consistent
reporting, plus labels/milestones/dependencies for planning.

## Final recommendation

If the team wants the shortest path to useful forecasting impact, the first executable branch should be:

1. stabilize the current forecasting substrate;
2. ship `mSSA`;
3. ship `regime_diagnostics`;
4. ship `HAVOK`;
5. only then promote Deep OKHS from experimental branch to forecasting API.

That ordering is the least speculative path and best matches both the current repository state and the uploaded roadmap.
