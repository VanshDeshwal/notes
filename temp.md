# Tie-break fix — cross-layer headline (mean pooling): 8.44 % → 7.82 % (0.084 → 0.078)

**Date:** 2026-06-22 · Applied **in place** to the latest canonical files:
`Chapters/7-results.tex`, `Chapters/8-discussion.tex`, `Chapters/9-conclusion.tex`.

> **Correction to my earlier change-map:** I had mislabelled the tables (off by one).
> The correct numbering is **7.1** res-split · **7.2** res-family · **7.3** res-main
> (pooling) · **7.4** res-layers (ablation) · **7.5** res-drift · **7.6** res-shortcuts ·
> **7.7** res-shap. So "the mean cell" lives in **Table 7.3**, not 7.2. **Table 7.2 (model
> family) correctly needs no edit** — which is exactly what you spotted. The cell edits
> themselves were correct; only my table numbers were wrong.

---

## The change

The cross-layer L2+L3+L4 MLP has four **mean-pooling** runs tied at val EER **2.83 %**;
the tie-break (lowest test HTER) selects the **7.82 %** run, not the **8.44 %** one. New
mean operating point: HTER **7.82 %** (0.078), FPR **2.90 %**, FNR **12.75 %**, AUC **0.943**,
confusion **349 / 134 / 4 / 51**. Attention (6.70 %) is unchanged. Only the L2+L3+L4 mean
cell propagates; all other ablation rows and the family/XGBoost/AE numbers are unaffected.

Note the **notation differs by chapter**: `7-results.tex` uses 2-dp percentages
(7.82 %, 12.75 %, 2.90 %); `8-discussion.tex` and `9-conclusion.tex` use decimal HTER
(0.078) and 1-dp percentages (12.8 %, 2.9 %). Edits below respect each file's style.

---

## `7-results.tex` — 11 edits (Table 7.3 / res-main was already 7.82, left as is)

| Where | Old → New |
|---|---|
| §7.3 prose (detection) | HTER 8.44→**7.82 %**, FPR 3.62→**2.90 %**, FNR 13.25→**12.75 %**, conf 347/133/5/53 → **349/134/4/51** |
| **Table 7.4** res-layers, L2+L3+L4 row | 8.44 %/3.62 %/13.25 % → **7.82 %/2.90 %/12.75 %** |
| §7.4 prose "deploys at 8.44 % (mean)" | → **7.82 %** |
| §7.4 prose "cuts FNR to 13.25 %… HTER 12.11 % to 8.44 %" | → **12.75 %… 7.82 %** |
| **Table 7.5** res-drift, Mean row | test FPR 3.62→**2.90 %**, test FNR 13.25→**12.75 %** |
| §7.5 prose "0.83 % to 13.25 %" | → **12.75 %** |
| §7.5 prose FNR range "13 %–24 %, FPR at most 7 %" | → **12.75 %–24 %, FPR at most 7.25 %** (the 7 % was already a slip — L2 is 7.25 %) |
| §7.6 prose "6.70 % to 8.44 %" | → **6.70 % to 7.82 %** |
| **Table 7.6** res-shortcuts, Mean row | 8.44 %/0.932 → **7.82 %/0.943** |
| §7.7 prose, composite vs mean | "essentially the same as mean (8.44 %), still behind attention" → "**close to but still behind mean (7.82 %), and further behind attention (6.70 %)**" |
| §7.8 prose "6.70 % to 8.44 %" | → **6.70 % to 7.82 %** |

## `8-discussion.tex` — 8 edits (decimal notation)

| Where | Old → New |
|---|---|
| §disc-endtoend | "deploys at 0.067 to 0.084" → **0.067 to 0.078** |
| §disc-layers | "FNR to 13.3 %… HTER 0.121 to 0.084" → **12.8 %… 0.078** |
| §disc-drift | "test FNR 13.3 %… FPR 4.8 % to 3.6 %" → **12.8 %… 4.8 % to 2.9 %** ("sixteen times" still holds: 12.8/0.8 = 16) |
| §disc-drift | "test HTER 0.067 to 0.084" → **0.067 to 0.078** |
| §disc-drift | "(HTER 0.067 against 0.084)" → **against 0.078** |
| §disc-protocol | "deploys at HTER 0.067 to 0.084" → **0.067 to 0.078** |
| §disc-composition | "matches mean (0.084), beaten only by attention" → "**comes close to mean (0.078) but is beaten by both poolings (0.078 mean, 0.067 attention)**" |
| §disc-ops | "misses more **and alarms less**" → "misses more, **at the same false-alarm rate**" (new mean FPR 2.9 % = attention 2.9 %) |

## `9-conclusion.tex` — 4 edits (decimal notation)

| Where | Old → New |
|---|---|
| §summary | "and 0.084 under mean pooling" → **0.078** |
| §summary | "HTER 0.067–0.084 against 0.126–0.149" → **0.067–0.078** |
| §summary | "(0.121 against 0.067–0.084)" → **0.067–0.078** |
| §summary | "ensemble 0.087, essentially matching mean (0.084) but not attention" → "**close to but short of mean (0.078) and short of attention (0.067)**" |

---

## One open decision — correlation filtering

The selected 7.82 % run uses **correlation filtering (CRS98)**; the 8.44 % run did not.
So two sentences are now in tension and I **left them untouched** for you to decide:

- `7-results.tex` §7.4: *"correlation filtering … never improved on the unfiltered model."*
- `8-discussion.tex` §disc-features: *"dropping one of each strongly correlated feature pair never helped …"*

Both were true when the mean headline was the unfiltered 8.44 % run. Options:

1. **Keep 7.82 %, rescope those two sentences** to *"never lowered the validation EER (the
   selection metric)"* — still true, and the deployed best happens to be CR-filtered only via
   the test-HTER tie-break. (The overall best deployer, attention 6.70 %, is unfiltered, so the
   main story is unaffected.)
2. **Restrict the tie-break to same-feature-engineering runs** — then the mean headline stays
   the unfiltered **8.44 %** and nothing about correlation filtering changes.

Tell me 1 or 2 and I'll finish it. Everything else is done and consistent.

---

## Verified
`grep`: no `8.44 / 0.084 / 13.25 / 13.3 % / 3.62 / 0.932 / "matches the monolithic" / "alarms
less"` remain. New values present in each file's own notation. res-main (Table 7.3) confirmed
already at 7.82 %/0.943 and left untouched. Arithmetic: (2.90+12.75)/2 = 7.825 %; 4+134 = 138
benign, 349+51 = 400 malicious, 538 total.

# Figure placement plan — precise (what · how to obtain · where)

**Date:** 2026-06-22. Line numbers are for the **current** files as delivered. Inserting a
figure shifts every later line down, so **work from the bottom of each file upward** (highest
line number first) and the rest stay valid.

Two source conventions:
- **Existing file** → a real `.svg`/`.png` already on disk; copy the `.pdf`/`.svg` into
  `Images/` (or a new `figures/`) and `\includegraphics` it. Paths are under
  `Code/outputs_new/models/MLP/`.
- **Generate** → a small bar chart that does not exist as a file; ~15 lines of matplotlib over
  numbers already in the chapter's own table (recipe in §4), or export the matching view from
  your dashboard.

**Run aliases** (the two new headline runs):
```
MEAN = Code/outputs_new/models/MLP/nmd_L2-3-4_MLP_all_PCAP_Seperate_split-by-pcap_HL3_LR0.001_BS128_OPTRMSprop_AGGmeanagg_ETL0__FE0_CRS98_L2.FE1_L3.FE1_L4.FE1_HD128/performance_metrics
ATTN = Code/outputs_new/models/MLP/nmd_L2-3-4_MLP_all_PCAP_Seperate_split-by-pcap_HL3_LR0.001_BS128_OPTRMSprop_AGGatnagg_ETL0__FE0_L2.FE1_L3.FE1_L4.FE1_HD128/performance_metrics
```
Each has `.svg` **and** `.png` of: `roc_curve`, `det_curve`, `confusion_matrix`,
`score_distribution`, `fpr_fnr_threshold`, plus a `shap/` folder (9 PCAPs × `L2/L3/L4/shap_group_importance.svg`).

---

## 1. Results chapter — `Chapters/7-results.tex` (the priority; currently 0 figures)

| ID | What the figure is | How to obtain it | Insert AFTER line |
|---|---|---|---|
| F5 | **Model-family bars** — grouped bars of val EER and test HTER for MLP / XGBoost / autoencoder. Shows the end-to-end gap at a glance. | Generate from Table 7.2 (3 rows). | **125** (after `\end{table}` of res-family) |
| F3a | **ROC curves, mean vs attention** — two ROC curves of the cross-layer detector side by side. | Existing: `MEAN/roc_curve.svg` + `ATTN/roc_curve.svg` (subfigure). | **179** (end of §7.3 prose) |
| F3b | **DET curve** — detection-error-tradeoff of the headline detector; the natural view for EER/HTER. | Existing: `ATTN/det_curve.svg` (best deployer). | **179** |
| F1 | **Layer-ablation bars** ⭐ — grouped bars, val EER vs test HTER across L2, L3, L4, L2+L3, L2+L4, L3+L4, L2+L3+L4. The chapter's central result; shows "+L4 raises val EER, lowers HTER." | Generate from Table 7.4 (7 rows). | **210** (after `\end{table}` of res-layers) |
| F2 | **Drift bars** ⭐ — grouped bars of val vs test FPR and FNR (mean + attention). Shows FNR jumping while FPR stays flat. | Generate from Table 7.5 (res-drift). Note: the headline runs have **no** temporal plot, so build this from the table. | **285** (after `\end{table}` of res-drift) |
| F6 | **Protocol bars** — chronological vs random HTER, same model. | Generate once the random split is run (Table 7.6 placeholder). | **339** (after `\end{table}` of res-shortcuts) |
| F7 | **Committee-progression bars** — HTER of frozen committee (0.116) → score combiner (0.110) → fine-tuned (0.091) → ensemble (0.087), with monolithic mean 0.078 / attention 0.067 as reference lines. | Generate from §7.7 prose numbers. | **388** (end of §7.7 prose) |
| F8 | **Autoencoder score distribution** — benign vs malicious score histograms for the novelty baseline; shows the supervision ceiling. | Existing: `Code/outputs_new/models/AUTOENCODER/<L4 run>/performance_metrics/score_distribution.svg`. | **411** (end of §7.8) |
| F4 | **3-panel per-layer SHAP** ⭐ — three bar panels (L2, L3, L4) of feature-group importance for the headline model. The text already has a TODO for exactly this. | Existing: `MEAN/shap/<pcap>/{L2,L3,L4}/shap_group_importance.svg` — pick one representative malicious capture (e.g. the Qakbot or Emotet capture) or average the 9. | **replace lines 458–459** (the `\verb` TODO) |
| F9 | **Attention-weights example** — per-flow attention inside one flagged capture ("which flow"). | Existing (not in headline run; use the sibling attention run): `…/nmd2_L2-3-4_MLP_…_AGGatnagg_…FE1_VT001_HD128/performance_metrics/shap/<pcap>/attention_weights.svg`. | **459** (just after F4) |

⭐ = highest value for breaking up the text.

### Paste-ready blocks (high-priority)

**F1 — after line 210:**
```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=0.9\textwidth]{Images/ablation_layers.pdf}
  \caption{Validation EER and deployed test HTER across feature-layer sets
           (common 323/538 split). Adding the Application layer (L4) raises the
           validation EER but lowers the deployed HTER.}
  \label{fig:res-ablation}
\end{figure}
```

**F4 — replace lines 458–459:**
```latex
\begin{figure}[ht]
  \centering
  \begin{subfigure}{0.32\textwidth}\includegraphics[width=\textwidth]{Images/shap_L2.pdf}\caption{Internet (L2)}\end{subfigure}\hfill
  \begin{subfigure}{0.32\textwidth}\includegraphics[width=\textwidth]{Images/shap_L3.pdf}\caption{Transport (L3)}\end{subfigure}\hfill
  \begin{subfigure}{0.32\textwidth}\includegraphics[width=\textwidth]{Images/shap_L4.pdf}\caption{Application (L4)}\end{subfigure}
  \caption{Feature-group importance (mean $|\text{SHAP}|$) per layer for the headline
           cross-layer model.}
  \label{fig:res-shap}
\end{figure}
```
(needs `\usepackage{subcaption}` in `thesis.tex`.)

**F3 — after line 179:**
```latex
\begin{figure}[ht]
  \centering
  \begin{subfigure}{0.48\textwidth}\includegraphics[width=\textwidth]{Images/roc_mean.pdf}\caption{Mean pooling}\end{subfigure}\hfill
  \begin{subfigure}{0.48\textwidth}\includegraphics[width=\textwidth]{Images/roc_attn.pdf}\caption{Attention pooling}\end{subfigure}
  \caption{ROC of the cross-layer MIL detector on the 538-capture test set.}
  \label{fig:res-roc}
\end{figure}
```

---

## 2. Cleanup — placeholder figures already in the text (just supply the image)

These are **commented-out** `\includegraphics` inside existing figure environments (they don't
error; they're empty placeholders the author left). Supply the image and uncomment:

| File · line | Placeholder | What to put there |
|---|---|---|
| `4-feature_extraction.tex` : **81** | `% …{figures/feature_pipeline}` | the feature-extraction pipeline diagram (or repoint to the existing `Images/Feature_Extraction_Pipeline.pdf` already used at line 171) |
| `5-feature_engineering.tex` : **144** | `% …{figures/fe_pipeline}` | the feature-engineering pipeline diagram |
| `appendix-2.tex` : **265** | `% …{figures/dashboard_overview}` | a dashboard screenshot (see §3) |

**Two architecture PDFs already exist but are unused** — add them where each baseline is
introduced (the MLP architectures sit at `6-framework.tex` lines 520 and 556):
- `Images/Autoencoder_Architecture.pdf` → a `\begin{figure}` after `6-framework.tex` line **565**.
- `Images/XGBoost_Architecture.pdf` → same area.

---

## 3. Appendix & dashboard (the appendix sections already exist)

`appendix-2.tex` already has: **Hyperparameter Grids** (line 189), **Extended Results**
(line 193), **Dashboard Screenshots** (line 197), Output Artefacts (199).

| Goes in | Content | Source |
|---|---|---|
| Extended Results (L193) | ROC/DET/PR/confusion for all 7 layer sets; training curves (`loss`, `eer`); calibration sweep `fpr_fnr_threshold.svg` | per-run `.svg` under `…/models/MLP/<run>/performance_metrics/` |
| Hyperparameter Grids (L189) | table of the search space (pooling, LR, FE, correlation filter, hidden dim, optimiser) | `results_analysis/all_results.csv` |
| Dashboard Screenshots (L197 → uncomment L265) | your dashboard views | **you provide** — drop into `Images/`, then point line 265 at it |

**Dashboard views → where** (export ≥150 dpi, crop the browser frame):
overview → appendix line 265; drift/temporal heatmap → §7.5 (alt to F2); per-PCAP attention
explorer → §7.9 or appendix; score/threshold explorer → §7.3 or appendix.

**External supplementary (not bound):** full `all_results.csv` (939 rows), the 13.8k raw
per-PCAP SHAP files, the code repo. Reference once in the appendix preamble.

---

## 4. Generation recipes (for the "generate" bars)

All four are grouped bar charts over numbers already in the chapter; ~15 lines of matplotlib,
or export the equivalent from your dashboard. Save as `Images/<name>.pdf`.

- **F1 ablation** — x = 7 layer sets; two bars each = val EER, test HTER (Table 7.4).
- **F2 drift** — x = {mean, attention}; bars = val FNR, test FNR, val FPR, test FPR (Table 7.5); the FNR pair towers over the FPR pair.
- **F5 family** — x = {MLP, XGBoost, AE}; two bars each = val EER, test HTER (Table 7.2).
- **F7 committee** — x = {frozen, combiner, fine-tuned, ensemble}; one HTER bar each (0.116, 0.110, 0.091, 0.087) + dashed reference lines at 0.078 (mean) and 0.067 (attention).

---

## Appendix vs supplementary (your earlier question)

Curated figures/tables → **appendix** (you already have the sections above); raw bulk
(`all_results.csv`, per-PCAP SHAP dumps, code) → **external supplementary**, referenced once.
