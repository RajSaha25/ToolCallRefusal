# Running the mech-interp notebooks on Google Colab

This walks through running the two notebooks on a Colab A100. The whole thing is "set one variable and
Run all" once the setup cell has done its job.

## What you need

- Colab with an **A100** runtime (Colab Pro / Pro+). A 14B model in bf16 needs ~28 GB; Colab's A100 is
  40 GB, which fits — but it's tight, so see the memory note below.
- Read access to the `RajSaha25/ToolCallRefusal` repo (it's private).
- A **GitHub token** with read access to the repo, saved as a Colab secret (one-time, below).

## One-time: add your GitHub token to Colab Secrets

1. In GitHub, create a fine-grained personal access token with **read** access to `ToolCallRefusal`
   (Settings → Developer settings → Fine-grained tokens).
2. In Colab, click the **key icon** in the left sidebar → **Add new secret**.
   - Name: `GITHUB_TOKEN`
   - Value: your token
   - Toggle **Notebook access** on.

You only do this once per Colab account; every notebook can then read the secret.

## Run notebook 1

1. Open `01_refusal_direction_and_suppression.ipynb` in Colab (File → Open notebook → GitHub tab →
   search `RajSaha25/ToolCallRefusal`; authorize GitHub if asked. Or download the `.ipynb` and upload it).
2. **Runtime → Change runtime type → A100 GPU.**
3. Run the **Setup (section 0)** cell first — it clones the repo, installs the missing packages, and
   moves into the project folder.
4. In the **Configuration** cell, set `MODEL_ID` to your model.
5. **Runtime → Run all.**

It extracts the refusal direction, validates it (ablation/addition), measures the tool-suppression
result, and saves the direction to `interp_artifacts/<model>/`. A few minutes plus short generation runs.

## Run notebook 2

Open `02_causal_followups_and_scaling.ipynb` **in the same Colab session** (don't disconnect after
notebook 1 — Colab storage is per-session, and notebook 2 loads the direction notebook 1 saved). Same
steps: run the setup cell, set the **same** `MODEL_ID`, Run all. This is the generation-heavy half
(~40 min on an A100 at the default sample sizes).

## Two cells worth checking on a new model

- **Notebook 1, section 3** confirms your model's chat template actually renders tool definitions. If it
  warns that no tool name appears, the tool-enabled modes won't mean anything for your model.
- **Notebook 2, section 2.1** prints raw generations next to the parsed tool calls. If the raw text
  clearly contains a tool call but `PARSED` is empty, add your model's tool-call syntax to
  `parse_tool_calls` before trusting the unsafe-rate numbers.

## Notes and troubleshooting

- **Out of memory** on the 40 GB A100? Lower the `N_*` sample sizes in the Configuration cell (and `BS`
  in notebook 2). The science still holds at smaller n; you just get wider confidence intervals.
- **Weights re-download each session.** Colab storage is ephemeral, so the model is fetched again on a
  fresh runtime. To avoid that, mount Google Drive and point `HF_HOME` at a Drive folder before the model
  loads (`os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"`).
- **Gated model?** If your `MODEL_ID` is gated on the Hugging Face hub, add an `HF_TOKEN` Colab secret and
  log in (`from huggingface_hub import login; login(userdata.get("HF_TOKEN"))`) before the model-load cell.
- **`GITHUB_TOKEN` not found.** Make sure the secret name matches exactly and *Notebook access* is on.

The deep explanation of the methods lives in `MECH_INTERP_GUIDE.md`; the results write-up is
`MECHANISTIC_INTERP.md`.
