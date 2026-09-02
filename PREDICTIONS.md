# Predictions vs outcomes, 2026-09-02 rerun

Predictions were written down before the GPU session (H100 80GB, pod naval_tomato_lark)
and are scored here against what came out. Confidence in parentheses is what was stated
beforehand. "Held / Missed / Partly" is the honest call, not a generous one.

| # | Prediction (before) | Outcome | Call |
|---|---|---|---|
| 1 | Patching: via-safe-call <= 10 points in every model; the rest of the flip is no-call. (70%) | Qwen 9 / Mistral 6 / Command-R 6 / Gemma 3 points via safe call; no-call carries 15 / 13 / 29 / 87. Unsafe-given-call after patching 0.89 / 0.93 / 0.92 / 0.80. | **Held** (4/4 so far; Llama pending) |
| 2 | Patching overall flip rates land near the published range (15-50%). | Qwen 24 (draft 18), Mistral 20 (24), Command-R 34 (62), Gemma 90 (46). | **Partly**: Command-R halved, Gemma doubled. Gemma's draft number came from a run with no tools in the prompt at all. |
| 3 | Command-R has the highest via-safe-call share. | Qwen does (9 vs 6). | **Missed** |
| 4 | Non-Gemma patching degeneracy < 5%; Gemma 10-30%. | Qwen 0, Mistral 0, Command-R 6, Gemma 8. | **Held** for three; Gemma lower than predicted |
| 5 | KL guardrail: Command-R, Mistral, Llama pass at some layer; Gemma fails everywhere; Qwen coin flip. (75%) | On generic harmless prompts (Arditi's protocol): Command-R passes (0.044; 8/17 layers pass; best L27 next to operating L26). Mistral fails at every layer (best 0.25; random 0.004). Gemma fails at every one of 32 layers (11.3; mean-out 2.5; random 0.04). On the dataset's own benign prompts nothing passes anywhere. Qwen/Llama pending. | **Partly**: Command-R and Gemma as predicted, Mistral missed |
| 6 | Steering rescoring under the v2 rule: Llama 20 and Mistral 17 become unusable; Mistral 11, Qwen 450, Gemma 12162, Command-R 49 hold. (80%) | Mistral 17 -> 36%, Qwen 700 -> 51%, Gemma addition -> 100%; the four named points stay <= 2%. Llama 20 stays at 6%. | **Held**, except Llama 20 did not become unusable |
| 7 | (not predicted) | Benign-side tool-calling collapses at the same coefficients as harmful-side calling: Mistral 0.51->0.05, Llama 0.93->0.12, Gemma 0.39->0.07, while text-judged over-refusal reads 0.08 / 0.41 / 0.14. | **Not predicted; changes the steering conclusion** |
| 8 | Gemma-27B mean-out ablation: coherent, KL 1-3, refusal flat-to-up. (Arditi-style drop is the ~20% outcome) | Coherent (1.3% degenerate vs 100% for the raw direction), KL 2.5 generic / 4.3 dataset. Refusal 0.66 -> **1.00** on 237 coherent outputs; the texts are fluent, on-topic refusals and prompts the baseline hedged on become flat refusals. | **Held** on direction; the size (+34 points) was beyond the predicted 0.60-0.80 |
| 9 | Random-direction control: within +/-3 points, 0% degeneracy for non-Gemma; Gemma 10-40% degenerate even from random. (80%) | Mistral -1 point, Command-R +0.5, Gemma +2; all three 0% degenerate. Gemma's random-direction KL is 0.04. | **Held** for the non-Gemma half; **missed** on Gemma being fragile to a random direction -- it isn't, at 27B |
| 10 | Mean-direction control breaks Gemma nearly completely, degrades the others moderately. (70%) | Gemma 100% degenerate (KL 46). Mistral: 10% degenerate, refusal +5. Command-R: 6% degenerate but refusal -20 (0.68->0.48) -- the mean direction carries part of Command-R's refusal signal (cos(r, mean) = 0.20). | **Held** for Gemma and Mistral; Command-R's drop was not predicted |
| 11 | Addition at 1x: coherent (>= 90%) in all five; benign refusal ~0-5% -> 10-35%; at the largest coherent coefficient 30-60%, never 100%. (75%) | Mistral: coherent at 1x but refusal 0.0 -> 0.004; 2x is the ceiling (5% degenerate) at 6-10%; 3x is 97% degenerate. Command-R: 1x is already 50% degenerate (41% refusal among the coherent half); 1.5x and above 100%. Gemma: coherent at 1x but refusal 0.013; 1.5x is 91% degenerate. | **Missed** in both directions: at 1x the effect is ~0 in Mistral and Gemma, and Command-R's coherence ceiling is below 1x. Coherent addition never exceeds ~40% in any model so far |
| 12 | Table 1 with the collaborator's scorer: unsafe columns rise ~10 points to the paper's numbers. | not rerun this session (CPU; needs his predicates applied to results/*.csv) | |

## What would have changed my mind (stated beforehand)

- Patching via-safe-call > 25% in any model: **did not happen** (max 9).
- Gemma mean-out ablation dropping refusal > 20 points coherently: pending.
- Addition at 1x reaching > 60% coherent refusal: pending.
