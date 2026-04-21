# Text2Image Prompts for Regime-Aware Forecasting Functions

## Purpose

These prompts are intended for text-to-image systems that generate developer-facing technical figures. The styling
target is:

`ICML-style, high-detail technical infographic. Style: modern scientific systems diagram, white background, clean vector layout, subtle blue-green-amber palette, readable labels, architecture review aesthetic.`

## Prompt 1. `analyze_regime_diagnostics(...)`

Use this prompt to visualize the internal logic
of [`analyze_regime_diagnostics(...)`](D:/data_old/WORK/Repo/Industiral/IndustrialTS/fedot_ind/core/models/ts_forecasting/regime_diagnostics.py).

```text
ICML-style, high-detail technical infographic. Style: modern scientific systems diagram, white background, clean vector layout, subtle blue-green-amber palette, readable labels, architecture review aesthetic. Show a function-level workflow diagram for a time-series regime analysis module named analyze_regime_diagnostics. Layout should be left-to-right with clearly separated labeled panels and thin arrows. Start with an input panel showing a one-dimensional training time series as a clean line chart. Then show a preprocessing panel: convert to numpy array, flatten to 1D, short-history guard branch, and adaptive window estimation using a helper estimate_window. From there branch into four analytical blocks in parallel. Block 1: autocorrelation analysis, with small lagged overlays and an ACF mini-plot, extracting acf_decay_rate and dominant_period. Block 2: spectral analysis, showing centered signal, FFT magnitude spectrum, and two derived indicators: spectral_concentration and spectral_flatness. Block 3: local linearity analysis, showing sliding local windows, simple line fits over segments, reconstruction residuals, and aggregation into local_linearity_score. Block 4: switching analysis, showing first differences, sign changes in gradients, and aggregation into switching_score. Then merge these signals into a compact typed record panel labeled RegimeDiagnosticsResult, listing series_length, dominant_period, acf_decay_rate, spectral_concentration, spectral_flatness, local_linearity_score, switching_score, regime_hint. On the far right, show a decision panel deriving regime_hint using threshold rules: periodic, switching, locally_linear, weak_structure, insufficient_history. Include small mathematical callouts, but keep them visually light: autocorrelation, FFT magnitude, geometric mean over arithmetic mean for spectral flatness, local MSE normalization, sign-change frequency. Use icon-free scientific notation and minimal typography. No photorealism, no glossy effects, no decorative background, no 3D rendering. The figure should feel like a conference-paper methods figure that explains both algorithmic stages and typed outputs.
```

## Prompt 2. `recommend_forecasting_model(...)`

Use this prompt to visualize the logic
of [`recommend_forecasting_model(...)`](D:/data_old/WORK/Repo/Industiral/IndustrialTS/fedot_ind/core/models/ts_forecasting/regime_routing.py).

```text
ICML-style, high-detail technical infographic. Style: modern scientific systems diagram, white background, clean vector layout, subtle blue-green-amber palette, readable labels, architecture review aesthetic. Create a technical routing diagram for a function named recommend_forecasting_model that takes a typed diagnostics object and produces a typed routing decision for forecasting models. Start on the left with an input card labeled RegimeDiagnosticsResult, listing compact fields: dominant_period, acf_decay_rate, spectral_concentration, spectral_flatness, local_linearity_score, switching_score, regime_hint, series_length. In the center show a rule-based routing engine with layered threshold gates. First gate: insufficient_history branch leading to naive_last_value and linear_trend fallback. Second gate: periodic signature detection using ACF decay, low spectral flatness, dominant period, and spectral concentration, routing to ssa_compat or mssa depending on period scale. Third gate: switching signature detection using switching_score, routing to havok. Fourth gate: local linearity branch using local_linearity_score, routing to okhs. Final fallback branch: weak structure routed to linear_trend or naive_last_value. Each branch should have a small confidence badge and concise rationale snippets like regime_hint=weak_structure, switching_score=0.45, dominant_period=24, spectral_flatness=0.17. On the right show an output typed object panel labeled RegimeRoutingDecision with fields primary_adapter, candidate_adapters, fallback_adapter, confidence, rationale, regime_hint. Include a lower band showing how this decision is stored as routing_recommendation metadata inside a forecasting benchmark run record, without forcing execution yet. Visually emphasize separation between diagnostics, recommendation, and execution. Use clear thin arrows, decision diamonds or threshold bars, compact table-like labels, and readable typography. No decorative gradients, no 3D objects, no UI chrome, no photorealistic elements. The figure should look like a polished ICML methods or systems diagram for an architecture review slide or paper appendix.
```

## Suggested Figure Set

If you want a multi-panel companion set instead of single images, split into:

- Panel A: diagnostics extraction pipeline
- Panel B: routing rule engine
- Panel C: integration inside run_forecasting_suite benchmark workflow
