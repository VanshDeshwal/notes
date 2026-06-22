# Tie-break fix — cross-layer headline (mean pooling): 8.44 % → 7.82 %

**Date:** 2026-06-22
**Files changed:** `Chapters/7-results_NEW.tex`, `Chapters/8-discussion_NEW.tex`,
`Chapters/9-conclusion_NEW.tex`, `thesis.tex` (includes rewired to the `_NEW` files).
Old chapter files left untouched as backup.

---

## 1. The rule and why the number changes

Selection metric is **validation EER at the operating threshold** (`val_eer_at_thr`).
Rule applied: **when two configurations tie on val EER to two decimal places (as a
percentage), break the tie by the lower test HTER.**

The cross-layer **L2+L3+L4 MLP** has **four mean-pooling runs tied at val EER
2.8263 % (→ 2.83 %)** on the common 323/538 set. Their test HTERs:

| Run (config) | val EER | test HTER | filter |
|---|---|---|---|
| `…meanagg…FE0_CRS98_L2.FE1_L3.FE1_L4.FE1` | 2.8263 % | **7.82 %** ← select | CRS98 |
| `…meanagg…FE1_CRS98` | 2.8263 % | 8.32 % | CRS98 |
| `…meanagg…FE1_VT001` (previously reported) | 2.8263 % | 8.44 % | none |
| `…meanagg…FE1` | 2.8263 % | 9.69 % | none |

The thesis reported the **8.44 %** run. The rule selects the **7.82 %** run.
Attention pooling (3.24 % / 6.70 %) is already the per-pooling best, so it is **unchanged**.

**New headline mean operating point** (from the 7.82 % run, common 538 test set):

| | test HTER | test FPR | test FNR | test AUC | TP / TN / FP / FN | val FPR | val FNR |
|---|---|---|---|---|---|---|---|
| old | 8.44 % | 3.62 % | 13.25 % | 0.932 | 347 / 133 / 5 / 53 | 4.82 % | 0.83 % |
| **new** | **7.82 %** | **2.90 %** | **12.75 %** | **0.943** | **349 / 134 / 4 / 51** | 4.82 % | 0.83 % |

Check: (2.90 + 12.75)/2 = 7.825 %; benign = 4+134 = 138; malicious = 349+51 = 400; total = 538. ✓
Validation side is identical between the two runs (same val EER, same val confusion), so the
drift table's val columns do **not** move.

**What did *not* change** (verified by re-deriving every cell with the tie-break):
every single-layer and two-layer row of the ablation already reported its tie-broken best;
the model-family table (MLP L2+L3 0.83 %/12.11 %; XGBoost 4.91 %/16.41 %; autoencoder
19.01 %/29.13 %) is unaffected. Only the L2+L3+L4 mean cell propagates.

Data source: `Code/outputs_new/models/MLP/…/performance_metrics/summary_metrics.json`
(225 unfiltered MLP runs on the common 323/538 set; both `nmd_` and `nmd2_` naming).

---

## 2. Every edit, by file

### `7-results_NEW.tex`
- **Table 7.2 (res-main), Mean row:** `8.44 % → 7.82 %`, AUC `0.932 → 0.943`.
- **§7.3 prose:** mean HTER `8.44 → 7.82`, FPR `3.62 → 2.90`, FNR `13.25 → 12.75`,
  confusion `347/133/5/53 → 349/134/4/51`.
- **Table 7.3 (res-layers), L2+L3+L4 row:** `8.44 % → 7.82 %`, FPR `3.62 → 2.90`, FNR `13.25 → 12.75`.
- **§7.4 prose:** "deploys at 8.44 % (mean)…" → 7.82 %; "cuts the FNR to 13.25 %… HTER from
  12.11 % to 8.44 %" → 12.75 % / 7.82 %.
- **Table 7.4 (res-drift), Mean row:** test FPR `3.62 → 2.90`, test FNR `13.25 → 12.75`.
- **§7.5 prose:** "0.83 % to 13.25 %" → 12.75 %; FNR range "between 13 % and 24 %" →
  "between 12.75 % and 24 %"; FPR bound "at most 7 %" → "at most 7.25 %" (corrects a
  pre-existing rounding slip — L2 FPR is 7.25 %).
- **§7.6:** "6.70 % to 8.44 %" → 6.70 % to 7.82 %; Table 7.5 (res-shortcuts) Mean row
  `8.44/0.932 → 7.82/0.943`.
- **§7.7 (experts):** see flag B below.
- **§7.8 (novelty):** "supervised model's 6.70 % to 8.44 %" → 6.70 % to 7.82 %.
- **§7.1 (setup), universe-comparison sentence:** see flag C below.
- **§7.4, correlation-filtering paragraph:** see flag A below.

### `8-discussion_NEW.tex`
- Six "6.70 % to 8.44 %" / "against 8.44 %" range mentions → 7.82 %.
- Drift paragraph: test FNR `13.25 → 12.75`, "about sixteen times higher" → "fifteen"
  (12.75 / 0.83 ≈ 15.4), FPR "4.82 % to 3.62 %" → "4.82 % to 2.90 %".
- Composite + ops sentences: flags B and D below.

### `9-conclusion_NEW.tex`
- "8.44 % under mean pooling" → 7.82 %; two "6.70 %–8.44 %" ranges → 6.70 %–7.82 %.
- Composite ensemble sentence: flag B below.

### `thesis.tex`
- `\include` switched from `7-results / 8-discussion / 9-conclusion` to the `_NEW` variants.

---

## 3. Judgment calls — please review

These are spots where a plain number swap would have made the text inconsistent, so the
wording was changed. Each is reversible.

**A. The headline mean model is now a correlation-filtered (CRS98) run.** The selected
7.82 % run uses correlation filtering; the old 8.44 % run did not. The prose previously
said correlation filtering "never improved on the unfiltered model" — now false at the
deployment level, since the CR run deploys better at the *same* val EER. I rescoped the
sentence to the selection metric: **"never lowered the validation EER, the metric used for
selection."** That is still literally true (CR never gave a lower val EER; it only wins the
test-HTER tie-break). **If you would rather the tie-break *not* cross feature-engineering
configs** (keep the headline an unfiltered model), the alternative is to restrict ties to
same-FE runs — in which case the mean headline stays **8.44 %**. Your call; I applied the
unrestricted rule as you stated it.

**B. The composite ensemble (8.72 %) no longer "essentially matches" mean.** With mean at
7.82 %, the 8.72 % ensemble is now behind *both* single-model poolings. Reworded in all three
chapters from "essentially matches / matches the monolithic model under mean pooling (8.44 %)"
to "comes close to but is behind the monolithic model under mean pooling (7.82 %)". The
ensemble's own number (8.72 %) is unchanged — it is a different model and outside the
monolithic tie-break. (If you want the same tie-break logic applied to the ensemble/expert
selections too, say so and I'll re-derive those from `advanced_training`/`ensemble_eval`.)

**C. Universe-comparison sentence (§7.1).** It illustrated how the smaller 530-capture
universe flatters results: "the same mean-pooling configuration reports 0.42 % / 5.51 %…
against 2.83 % and 8.44 % on the full set." The 0.42 %/5.51 % run is a specific pre-bugfix
FE1 run (`MLP_Bugged/…meanagg…FE1`); its exact bug-fixed 538 twin is the 9.69 % run, not
8.44 %. To keep the sentence consistent with the new headline I changed it to "the
cross-layer mean-pooling **detector** … against 2.83 % and **7.82 %**" and dropped the
"same configuration" claim (it was already loose). The pedagogical point is intact.

**D. "Misses more and alarms less" (§Discussion-ops).** The new mean run has FPR 2.90 % —
**equal** to attention's 2.90 % (both FP = 4/138). So mean no longer "alarms less"; it alarms
the same. Changed to **"misses more, at the same false-alarm rate."** (As a bonus this fixes a
pre-existing slip: the *old* mean FPR was 3.62 %, i.e. *higher* than attention, so "alarms
less" was already wrong.)

---

## 4. Verification done

- `grep`: no stray `8.44 / 13.25 / 3.62 % / 0.932 / 347 / "alarms less" / "sixteen times" /
  "never improved on the unfiltered" / "essentially"` remain in any `.tex`.
- No headline numbers are cited in chapters 1–6 or the abstract, so nothing else to sync.
- `pdflatex` full build (2 passes, draftmode): **0 fatal errors, 0 undefined cross-references**
  (the 173 undefined *citations* are only because bibtex was not run in the test).
- Arithmetic re-checked (HTER, confusion totals, drift multiplier).

# Figure & diagram placement plan

**Date:** 2026-06-22 · Scope: *placement map only* (you insert the figures).
Goal: break up the text — especially **Chapter 7 (Results), which currently has zero
figures** — with relevant, already-available plots, and decide what goes in the appendix.

---

## 0. Your question: appendix or supplementary material?

**Recommendation — use the appendix for curated extras; reserve external "supplementary
material" only for bulk artifacts.** Reasons:

- A master's thesis is one bound, self-citable document. An examiner expects to find a
  referenced figure/table in **Appendix X**, not in a separate file. You already have
  `appendix-1` and `appendix-2`, so the home is established.
- "Supplementary material" is a journal/online convention. It makes sense only for things
  that genuinely cannot sit in a PDF: the **full 225-run results CSV**, the **13,874 raw
  per-PCAP SHAP files**, the **code repository**, and the `.svg` sources. Point to these as
  an external package / repository link.

So: **curated figures and tables → appendix** (referenced from the chapters); **raw bulk →
external supplementary**, mentioned once in the appendix preamble.

---

## 1. Highest-value additions (do these first)

Chapter 7 is the wall of text. These five figures carry the most narrative weight:

| # | Section | Figure | Why it matters |
|---|---|---|---|
| F1 | §7.4 Ablation | **Layer-set bar chart** — val EER vs test HTER across L2, L3, L4, L2+L3, L2+L4, L3+L4, L2+L3+L4 | This is the central result. The "L4 raises val EER but lowers HTER" story is far clearer as bars than prose. |
| F2 | §7.5 Drift | **Val-vs-test FPR/FNR bars** (mean & attention) | Shows FNR exploding while FPR stays flat — the drift-on-the-malicious-side claim, visualized. |
| F3 | §7.3 Detection | **ROC + DET overlay**, mean vs attention | The "how good is it" visual; DET is the natural view for EER/HTER. |
| F4 | §7.9 Explainability | **3-panel per-layer SHAP group importance** (L2/L3/L4) | The chapter text already has a `% [FIGURE …]` TODO for exactly this. |
| F5 | §7.2 Model family | **Family comparison bars** (MLP vs XGBoost vs autoencoder) | Makes the end-to-end-wins gap immediate. |

F1, F2, F5 are **grouped bar charts that do not yet exist as single files** — they need a
~30-line matplotlib script over the numbers already in `results_analysis/*.csv` (or your
dashboard can export them). F3 and F4 already exist as files (paths below).

---

## 2. Full section-by-section map

Run-folder aliases (under `Code/outputs_new/models/MLP/`):

- **MEAN_RUN** = `nmd_L2-3-4_MLP_all_PCAP_Seperate_split-by-pcap_HL3_LR0.001_BS128_OPTRMSprop_AGGmeanagg_ETL0__FE0_CRS98_L2.FE1_L3.FE1_L4.FE1_HD128/performance_metrics`
- **ATTN_RUN** = `nmd_L2-3-4_MLP_all_PCAP_Seperate_split-by-pcap_HL3_LR0.001_BS128_OPTRMSprop_AGGatnagg_ETL0__FE0_L2.FE1_L3.FE1_L4.FE1_HD128/performance_metrics`

(These are the two **new headline** runs — mean 7.82 %, attention 6.70 %. Every per-run
plot below exists as both `.png` and `.svg`; use `.svg` or `.pdf` for print quality.)

| Section | Figure | Source | Suggested caption | Priority |
|---|---|---|---|---|
| §3 Background (metrics) | DET-curve schematic to illustrate EER/HTER | `ATTN_RUN/det_curve.svg` (as illustration) or a hand-drawn schematic | "DET curve and the equal-error operating point." | optional |
| §4 Feature extraction | *(fix broken ref — see §3 below)* | `Images/Feature_Extraction_Pipeline.pdf` | — | cleanup |
| §5 Feature engineering | TCP-state-machine / engineered-feature diagram | **generate** (or dashboard) | "Connection-state features derived from the TCP flag sequence." | medium |
| §6 Framework | Autoencoder architecture | `Images/Autoencoder_Architecture.pdf` *(exists, currently unused)* | "Autoencoder baseline architecture." | easy win |
| §6 Framework | XGBoost-head architecture | `Images/XGBoost_Architecture.pdf` *(exists, currently unused)* | "XGBoost classification head on the flow encoder." | easy win |
| **§7.2 Family** | **F5** family comparison bars | **generate** from `results_analysis/baselines.csv` | "Best validation-selected configuration per model family (common 323/538 set)." | high |
| **§7.3 Detection** | **F3** ROC + DET, mean vs attention | `MEAN_RUN/roc_curve.svg` + `ATTN_RUN/roc_curve.svg`; `…/det_curve.svg` | "Cross-layer detector under mean and attention pooling." | high |
| §7.3 Detection | Score distribution (benign vs malicious) | `ATTN_RUN/score_distribution.svg` | "Test-set score separation, attention pooling." | medium |
| **§7.4 Ablation** | **F1** layer-set val-EER/HTER bars | **generate** from `results_analysis/advanced_eval.csv` | "Validation EER and deployed HTER across feature-layer sets." | **highest** |
| **§7.5 Drift** | **F2** val-vs-test FPR/FNR bars | **generate** (numbers in Table 7.4) | "Validation vs test error: the malicious-side rate (FNR) rises; the benign-side rate (FPR) does not." | **highest** |
| §7.5 Drift | Temporal drift panel | `MEAN_RUN/` → `temporal_fpr_fnr.svg`, `temporal_metrics_heatmap.svg` | "Per-period error on the test timeline." | medium (or dashboard) |
| §7.6 Protocol | Chronological vs random HTER bars | **generate** once random split is run (Table 7.5 placeholder) | "The same model under two evaluation protocols." | blocked on re-run |
| §7.7 Experts | Committee → combiner → ensemble HTER bars | **generate** from `results_analysis/advanced_training.csv` | "Deployed error as the committee is trained toward end-to-end." | medium |
| §7.8 Novelty | Autoencoder score distribution / ROC | autoencoder headline run `…/score_distribution.svg` | "Benign-trained autoencoder: the supervision ceiling." | medium |
| **§7.9 Explainability** | **F4** 3-panel per-layer SHAP | `MEAN_RUN/shap/<pcap>/{L2,L3,L4}/shap_group_importance.svg`, aggregated across PCAPs | "Feature-group importance by layer (mean \|SHAP\|)." | high (TODO already in text) |
| §7.9 Explainability | Attention-weights example | `MEAN_RUN/attention_weights.svg` | "Per-flow attention within one flagged capture." | medium |
| §7.10 Deployment | Scoring/update path schematic | **generate** (or dashboard) | "Scoring a new capture and warm-start updating from a saved bundle." | optional |

**Discussion (Ch 8):** keep it prose-driven; reference the Results figures rather than
duplicating. At most one conceptual drift-mechanism schematic if you want a visual anchor.
**Conclusion (Ch 9):** no figures (convention).

---

## 3. Existing-asset cleanup (independent of the above)

Three `\includegraphics` point at files that **do not exist** — they will error or show
"file not found" boxes on a clean build:

- `Chapters/4-feature_extraction.tex:81` → `figures/feature_pipeline` (probably a duplicate
  of `Images/Feature_Extraction_Pipeline.pdf` — repoint or supply the file).
- `Chapters/5-feature_engineering.tex:144` → `figures/fe_pipeline` (missing — supply or remove).
- `appendix-2.tex:265` → `figures/dashboard_overview` (missing — this is a **dashboard
  slot**, see below).

Two architecture PDFs already exist but are **never used**: `Autoencoder_Architecture.pdf`,
`XGBoost_Architecture.pdf` → place them in §6 as noted (easy wins).

There is no `figures/` directory in the thesis folder; the broken refs expect one. Either
create `figures/` or repoint these to `Images/`.

---

## 4. Dashboard figures (you will provide)

I could not find dashboard exports in the workspace. When you drop them in, here is where
each belongs. Export at ≥150 dpi (PNG) or SVG, crop the browser chrome:

| Dashboard view | Placement | Replaces / fills |
|---|---|---|
| Overview / landing page | Appendix | the broken `figures/dashboard_overview` ref in `appendix-2.tex:265` |
| Temporal drift heatmap / timeline | §7.5 (alternative to generated F2/temporal panel) | — |
| Per-PCAP attention explorer (one flagged capture) | §7.9 or appendix | complements F4 |
| Score-distribution / threshold explorer | §7.3 or appendix | complements F3 |
| Layer-ablation comparison view | §7.4 (alternative to generated F1) | — |

Tell me the filenames once they're in `Images/` (or a new `figures/`) and I can give exact
`\begin{figure}` blocks with captions, labels, and sizing.

---

## 5. Appendix structure (curated extras)

Suggested appendix sections, all sourced from files that already exist:

1. **Full evaluation curves** — ROC/DET/PR + confusion matrix for all seven layer sets.
2. **Training diagnostics** — `loss.svg`, `eer.svg`, `learning_rate.svg`, `gradient_norm.svg`
   for the two headline runs.
3. **Extended explainability** — per-layer SHAP for several PCACPs, `shap_importance_by_outcome`,
   more attention-weight examples.
4. **Hyperparameter grid** — table of the search space (pooling, LR, FE, correlation filter,
   hidden dim, optimizer) and selected values.
5. **Expert-combination architecture + full results table** (the committee diagram + the
   numbers behind §7.7).
6. **Threshold/calibration sweep** — `fpr_fnr_threshold.svg` for the headline model.

**External supplementary (linked, not bound):** full `all_results.csv` (939 rows), the raw
per-PCAP SHAP dumps, and the code repository.

> Note: the Linux build sandbox briefly held a truncated copy of `thesis.tex` (last line
> `\e` instead of `\end{document}`). Your actual file is intact — this was a sandbox-sync
> artifact only, confirmed against the authoritative file view.
