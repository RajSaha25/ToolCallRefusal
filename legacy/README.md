# Legacy notebooks

These are earlier, Qwen3-14B-specific notebooks, kept for reference. They are superseded by the
model-agnostic notebooks in the repo root:

- `01_refusal_direction_and_suppression.ipynb`
- `02_causal_followups_and_scaling.ipynb`

What is here:
- `Mechanistic_Refusal_Direction.ipynb` — the original round-1 mech-interp notebook (extract direction,
  validate, projection by mode). Replaced by notebook 01.
- `Tool_Refusal_Mechanistic_Interp.ipynb` — an earlier attempt at a single consolidated mech-interp
  notebook, built mostly as a results viewer. Replaced by notebooks 01 + 02.
- `Copy_of_Behavioral_eval.ipynb` — a duplicate of the behavioral eval notebook (which remains in the
  repo root as `Behavioral_eval.ipynb`).


## Legacy scripts

Earlier iterative mech-interp scripts, Qwen3-14B-specific, superseded by the headless runners in the
repo root (`run_direction_and_suppression.py`, `run_scaled_evaluation.py`) and the notebooks. Kept for
provenance — they record the research path, including a couple of dead-ends later corrected.

- `round2_experiments.py` — system-condition interaction, suppression heads/MLPs, a first activation-
  patching attempt (null, floor effect), and an initial steering sweep at too small a coefficient scale.
- `round2_steering_fix.py` — re-ran steering at the correct scale and added the behavioral base-rate
  diagnostic that surfaced the scorer under-counting bug.
- `tier1_followups.py` — same-direction cosine, patching redo, and AUC, all on the fixed scorer.

They load the saved direction from `interp_artifacts/refusal_dirs.pt`; run
`run_direction_and_suppression.py` first to produce it.
