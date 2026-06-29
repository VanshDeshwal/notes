# Response to Supervisor Feedback — Meeting 2026-06-29

**Thesis:** *Enhanced Malware Detection by Analysing Network Traffic Through AI*
**Prepared:** 2026-06-27

> **APPLIED (2026-06-27):** All edits in Parts A–D and E2 below have been written into the thesis source files. Build verified — `pdflatex` runs with **0 fatal errors and 0 undefined references** (26 pre-existing "overfull hbox" cosmetics remain). **Three things were deliberately left:**
> 1. **§8.7 / Part E1** — the advanced-training inconsistency, left for your decision on the canonical architecture (its decimals are also left un-converted until then).
> 2. **§6.1 "provenance"** (Part C, optional) — left as-is; the reviewer complimented the word, so changing it on an optional note seemed wrong.
> 3. **Figure 6.2** (Part D) — the "Flow N / h_n" mismatch is inside the binary `Images/MLP_Architecture.pdf`; it must be regenerated in your diagram tool (there is no editable source in the repo). No `.tex` change needed.
>
> This document is kept as the record of what changed and why, and as the answers to defend at the meeting.

---

## How to use this document

The review is overwhelmingly positive ("Good job" on the large majority of sections). The points that need *action* fall into five buckets:

1. **Substantive questions** the supervisor wants answered in the text (XGBoost untrained encoder; the model-selection "lesson"; the application-layer explanation; the weak-label framing). These are in **Part A** with a defended answer + the exact edit.
2. **One notation pass** — convert the Discussion's decimal HTER/EER figures to percentages with two decimals (**Part B**). Results already uses percentages, so this is purely making the Discussion match.
3. **Wording / typography fixes** — small, mechanical (**Part C**).
4. **One figure fix** — Figure 6.2 (**Part D**).
5. **Two inconsistencies I found that the review did *not* flag** but that violate the "100% correct / consistent" bar (**Part E**). Read these — one is important.

Every edit is given as `OLD` (exact current text, so you can find-and-replace) → `NEW`. Line numbers are from the current chapter files. **Part F** lists every "Good job" item so nothing in the review is left unaccounted for.

Two facts worth carrying into the meeting:

- **The XGBoost answer is verified against your code** (`lib/libai/libxgboost.py`, branch `BugFix`): the function `xgboost_wpcap_based` builds a fresh transformer with `build_pcap_transformer(...)` and there is **no `backward()`, optimizer, or `optim.` call anywhere in the module** — only `xgb.train(...)` runs. So the encoder is provably never trained. Both reasons the supervisor guesses are correct and they compound. Details in A1.
- On the §8.1 directionality: **your thesis is correct** ("adding L4 *raises* validation EER but *lowers* deployed error"). The reviewer's own note paraphrases it the other way round ("increases Validation accuracy, but decreases Test accuracy"), which is itself the signal that the sentence is easy to misread. The A3 rewrite states the direction in plain better/worse terms so it can't be flipped.

---

# Part A — Substantive points (need a defended answer)

## A1. Why is the XGBoost encoder never trained? (§7.2, repeated at §8.8)

> **Reviewer (§7.2):** "…the per-flow encoder is randomly initialised and never trained… this row measures a tree classifier on an untrained representation rather than trees against neural networks in general. This should require further explanation. Why are they untrained… Is it due to the weak labeling? Is it due to the inability to backpropagate the attention-dependent gradients through the tree? If these are the reasons, it should be stated here."
>
> **Reviewer (§8.8):** "Why is not the XGBoost classifier actually trained? This needs an answer here. Is it that the flows are not well labelled, so no label is provided? Justify."

**Answer (verified in code — both of the supervisor's hypotheses are correct, and they reinforce each other):**

1. **Non-differentiable head → no gradient path into the encoder.** The MLP detector is trained end-to-end: the bag-level binary-cross-entropy loss back-propagates through the linear classifier *and the attention pooling* into the per-flow encoder, so the representation is shaped by the detection objective. Gradient-boosted trees are not differentiable in that way — there is no gradient to push from the XGBoost objective back through the attention pooling into the encoder. So even though XGBoost itself is trained (the trees are fit), the *encoder feeding it cannot be trained by the same signal*.

2. **Weak labels → no per-flow target to train the encoder another way.** The only labels available are at the PCAP/bag level (the whole point of the MIL framing, Ch. 6). There are no per-flow labels, so the encoder cannot be trained directly at flow level either. The single supervisory signal is the bag label, and the only way to propagate it into the encoder is a differentiable head — which the trees are not.

The two compound: with a differentiable head (MLP) the weak bag label *can* train the encoder; with trees it cannot, and there is no per-flow label to fall back on. The encoder therefore stays at its random initialisation and XGBoost classifies a fixed random projection. This makes the row an honest ablation of *trained vs. untrained representation*, not *trees vs. neural networks*.

**Code evidence (branch `BugFix`, `lib/libai/libxgboost.py`, `xgboost_wpcap_based`):** the transformer is built fresh (`build_pcap_transformer(len(feature_cols), hidden_dim, …)`), embeddings are extracted with `get_pcap_embeddings(transformer, …)` in a forward-only pass, then `xgb.train(...)` fits the trees on those embeddings. A grep of the whole module for `backward` / `.step()` / `optim.` returns **0 matches**. (A `warmstart_path` exists to *load* a pre-trained transformer, but in the reported model-family experiment it is unused, so the encoder is random.)

**Suggested edit — §7.2** (`Chapters/7-results.tex`, after the sentence ending "…rather than trees against neural networks in general.", line ~96):

```
OLD:  In the XGBoost head the per-flow encoder is randomly initialised and never
      trained, so the trees classify a fixed random projection of the flow features:
      this row measures a tree classifier on an \emph{untrained} representation rather
      than trees against neural networks in general.

NEW:  In the XGBoost head the per-flow encoder is randomly initialised and never
      trained, so the trees classify a fixed random projection of the flow features:
      this row measures a tree classifier on an \emph{untrained} representation rather
      than trees against neural networks in general. The encoder is untrained for two
      reasons that compound. First, the labels are reliable only at the capture level
      (the weak-label setting of Chapter~\ref{cha:framework}), so there is no
      per-flow target with which to train a flow encoder directly. Second, the only
      available signal is the bag label, and gradient-boosted trees are not
      differentiable, so --- unlike the MLP, whose bag loss back-propagates through the
      pooling into the encoder --- the XGBoost objective cannot train the representation
      that feeds it. The encoder therefore stays at initialisation. Boosted trees on a
      \emph{trained} encoder's embeddings would be the fair trees-vs-network comparison
      and were not run.
```

*(If you prefer a tighter pointer than the chapter, use the label on §6.3.1 "The Weak-Label Problem in PCAP-Labelled Datasets".)*

**Suggested edit — §8.8** (`Chapters/8-discussion.tex`, line ~256). The Discussion already states the *what*; add a one-line *why* and point back to §7.2 so the two agree:

```
OLD:  The XGBoost head classifies on a flow encoder that is randomly initialised and
      never trained, so it sees a fixed random projection of the features.

NEW:  The XGBoost head classifies on a flow encoder that is randomly initialised and
      never trained, so it sees a fixed random projection of the features. It is
      untrained because the trees are not differentiable and the labels are only at the
      bag level (Section~\ref{sec:res-family}): there is no per-flow label to train the
      encoder, and no gradient path from the tree objective back into it.
```

---

## A2. The model-selection "lesson" reads as cheating (§8.3) — **the most important point in the review**

> **Reviewer (§8.3), on "…the full-stack model should be the deployment choice even though a smaller model wins on validation":** "I wish it were that easy. This would be cheating, going against our own methodological choices. Perhaps you can say that the Validation-driven model selection methodology, [while] it turned out useful to detect this behaviour, it would not suffice by itself to fight against concept drift, which would require a finer split (using a test set prior to the deployment) in order to make concept-drift aware a priori model selection."

**He is right, and this needs to change.** The thesis's own rule is "nothing is chosen on the test set." But "deploy the full model even though it loses on validation" can only be justified by looking at the test result — at real deployment you have *only* the validation signal, and on that signal the application layer looks harmful and you would pick L2+L3. Recommending the full model therefore smuggles in test knowledge. The honest, and actually stronger, claim is: validation-driven selection was enough to **detect** the disagreement (post hoc), but it cannot **fix** it — it cannot pick the drift-robust model a priori, because the evidence for the application layer is exactly the drift the validation set can't see. Selecting for that would need a finer split: a second, chronologically *later* hold-out placed before deployment, against which selection can be made drift-aware.

**Suggested edit — §8.3** (`Chapters/8-discussion.tex`, lines ~87–91):

```
OLD:  The lesson is that the validation EER, used alone, can prefer the wrong model,
      and that the full-stack model should be the deployment choice even though a
      smaller model wins on validation. This is the same blindness discussed in
      Section~\ref{sec:disc-drift}, seen from the angle of model selection.

NEW:  The lesson is about the limits of the selection rule, and it has to be stated
      carefully so as not to cheat. The validation EER, used alone, prefers the smaller
      L2+L3 model; that the full L2+L3+L4 model in fact deploys better is something only
      the test set reveals, and the test set plays no part in selection
      (Section~\ref{sec:res-setup}). It would therefore be circular to conclude "deploy
      the full-stack model": at deployment time only the validation signal is available,
      and on that signal alone the application layer looks harmful. The honest conclusion
      is narrower and more useful. Validation-driven selection was enough to \emph{detect}
      this behaviour --- to show that the validation EER and the deployed error can
      disagree, and why --- but it does not by itself \emph{fix} it: it cannot choose the
      drift-robust model in advance, because the evidence that the application layer pays
      off is precisely the concept drift that the validation set, drawn from near the
      training period, cannot contain. Making model selection concept-drift-aware would
      require a finer split --- a second, chronologically later hold-out placed before
      deployment --- against which the choice could be priced. This is the same blindness
      discussed in Section~\ref{sec:disc-drift}, seen from the angle of model selection,
      and it is taken up as future work in Section~\ref{sec:concl-future}.
```

**Consistency check for you:** make sure the Conclusion's central-finding paragraph (Ch. 9, which the reviewer liked) does **not** still say "deploy the full model." If it implies that, soften it to match the above ("validation selection detects but cannot pre-empt drift"). The reviewer didn't flag Ch. 9, but if the two disagree it will be noticed.

---

## A3. The application-layer explanation is "a bit weak" + the directionality trap (§8.1)

> **Reviewer (§8.1):** "Application layer increases Validation accuracy, but decreases Test accuracy, which is a sign that the validation metric does not see what the application layer is good for. This explanation seems a bit weak. It needs some further discussion."

Two things. (a) Your text is **correct** and the reviewer's paraphrase is inverted (he wrote "increases Validation accuracy / decreases Test accuracy"; your thesis says L4 *raises validation EER* = worse validation, and *lowers deployed error* = better test). That inversion is the tell that the sentence is hard to track, because the reader has to remember that higher EER = worse. (b) The fix is to (i) state the direction in plain "worse/better" language so it can't be flipped, and (ii) give a one-line mechanism and an explicit pointer to §8.3/§8.5 so "weak" becomes "summarised here, argued there."

**Suggested edit — §8.1** (`Chapters/8-discussion.tex`, lines ~18–21):

```
OLD:  Second, adding the application layer raises the
      validation EER but lowers the deployed error, which is a sign that the
      validation metric does not see what the application layer is good for
      (Section~\ref{sec:disc-layers}).

NEW:  Second, adding the application layer makes the model look \emph{worse} on
      validation (a higher validation EER) yet perform \emph{better} in deployment (a
      lower test error). The application layer earns its place precisely on the
      malicious traffic that has drifted away from the training period --- traffic the
      validation set, drawn from near that period, does not contain --- so the validation
      metric cannot reward it. The mechanism, with the numbers, is in
      Section~\ref{sec:disc-layers}, and the drift it depends on is quantified in
      Section~\ref{sec:disc-drift}.
```

---

## A4. "frozen" is overloaded (§7.9, and the sub-point under §8.1)

> **Reviewer (§7.9):** "the use of the term *frozen* is used before … more precisely, and with a different meaning, so I would avoid using it here with such a different purpose."
> **Reviewer (§8.1):** "What do you mean with *frozen model*? The methodology always trains in Train, establishes thresholds in Validation, and measures performance in Test, so this is confusing."

**Answer:** "frozen" is used in two senses: (a) **frozen experts / frozen committee** — encoder weights not retrained, in the composition sections; this is the precise, standard usage and should stay. (b) **frozen threshold / frozen model** — the decision threshold fixed on validation; this collides with (a) and is what confuses the reader. Fix: use **"fixed"** for the threshold sense everywhere (the abstract and intro already say "fixed on validation", so this is also the more consistent word), and reserve **"frozen"** for the experts.

**Edits — keep "frozen" (experts), do NOT change:** `7-results.tex` lines 284, 298, 302, 322, 335, 338, 343, 347; `8-discussion.tex` lines 222, 226, 227, 237, 246.

**Edits — change "frozen" → "fixed" (threshold/model sense):**

| File / line | OLD | NEW |
|---|---|---|
| `7-results.tex` ~25 | "the test HTER at that **frozen** threshold" | "the test HTER at that **fixed** threshold" |
| `7-results.tex` ~125 | "FPR and FNR are at the **frozen** threshold." | "FPR and FNR are at the **fixed** threshold." |
| `7-results.tex` ~158 | "FPR and FNR are at the **frozen** threshold." | "FPR and FNR are at the **fixed** threshold." |
| `7-results.tex` ~443 | "threshold **frozen** on validation" | "threshold **fixed** on validation" |
| `8-discussion.tex` ~21 | "when the **frozen model** is run on the test set" | "when the **trained model** is run on the test set at the threshold fixed on validation" |
| `8-discussion.tex` ~162 | "the **frozen** threshold lets more of them through" | "the **fixed** threshold lets more of them through" |
| `8-discussion.tex` ~178 | "so the **frozen** threshold transfers better" | "so the **fixed** threshold transfers better" |

---

## A5. Shuffling validation/test "seems more aesthetics than practical" (§6.2.3)

> **Reviewer:** "the random shuffling of validation and test set seems more aesthetics than practical."

**Answer:** He is half-right, and the text can say so plainly. The **headline metrics (EER, HTER) are computed over the whole val/test set and are order-independent — shuffling cannot and does not change them.** So there is no methodological sleight here. The shuffle matters only for *order-sensitive* computations: running-average curves and any *subsample* drawn from val/test (e.g. the SHAP background of 100 flows), which on a time-ordered split would be biased toward the earliest captures. The chronology guarantee that actually matters lives in the split *boundaries*, which are fixed before any shuffle. Saying this converts "aesthetics" into a stated, bounded, practical reason.

**Suggested edit — §6.2.3** (`Chapters/6-framework.tex`, append to the paragraph ending "…before shuffling occurs.", line ~194):

```
ADD:  It is worth being explicit about what this does and does not affect: the
      set-level metrics this thesis reports (EER and HTER, computed over the whole
      validation or test set) are independent of capture order, so shuffling never
      changes a reported number. Its only effect is on order-sensitive quantities ---
      running-average diagnostics, and any subsample drawn from a split (such as the
      SHAP background of Section~\ref{sec:res-explain}) --- which on a time-ordered set
      would otherwise be biased toward the earliest captures. The chronological
      guarantee is carried entirely by the split boundaries, which are fixed before the
      shuffle.
```

---

## A6. Presence-flag / packet-count for the second protocol group — "literature or well-founded source?" (§4.6)

> **Reviewer:** "…represented more economically by a presence flag and a packet count only, on the grounds that their mere appearance in a flow is informative… Is this based on any literature finding or well-founded source? Please, clarify."

**Answer (be honest — it is an engineering decision, not a cited result):** these protocols (IPP, IRC, LDAP, MGCP, …) are rare in the corpus, so detailed per-protocol statistics would be overwhelmingly zero and sparse — they add dimensionality without reliable signal. The *presence* of some of them is occasionally telling (IRC as a botnet command-and-control channel is the textbook case), so a presence flag + count keeps the cheap signal and drops the unreliable detail. State it as a deliberate dimensionality/sparsity trade-off; optionally cite the IRC-C2 point if you have a reference in Ch. 2.

**Suggested edit — §4.6** (`Chapters/4-feature_extraction.tex`, after "…even when their fine-grained statistics are not.", line ~411):

```
ADD:  This is a deliberate design economy rather than a result imported from the
      literature. These protocols occur rarely in the corpus, so their fine-grained
      per-protocol statistics are sparse and mostly zero --- dimensionality without
      dependable signal --- whereas the bare fact that a flow uses one of them is
      occasionally diagnostic; the use of IRC for botnet command-and-control is the
      classic example. The presence flag and packet count keep that signal at negligible
      cost.
```

*(If you have an IRC/botnet citation in Ch. 2, append it after "classic example".)*

---

## A7. "The seven steps" — the reviewer counted six (§5.2)

> **Reviewer:** "*The seven steps*: I could only count six steps. Are you missing one or did I miss something?"

**Answer: there are genuinely seven** — they are written as one long semicolon-separated sentence, which is exactly why they are easy to under-count (and the same sentence is the source of the "first line does not fit within margins" comment). Fix both at once by making the seven an explicit numbered list:

1. read the per-packet local table `l{layer}_all_pcaps.parquet`;
2. call the layer's engineering function to produce a per-flow table of engineered columns;
3. write that table on its own to `l{layer}_all_fe.parquet`;
4. read the per-flow global table `l{layer}_all_gf.parquet`;
5. left-merge the engineered columns onto it on the flow key;
6. fill any engineered column left unmatched by the merge with zero;
7. write the result to `l{layer}_all_gf_wfe.parquet`.

**Suggested edit — §5.2** (`Chapters/5-feature_engineering.tex`, replace the run-on sentence at lines ~130–138):

```
OLD:  For a given layer, the pipeline executes a fixed sequence. It reads the
      per-packet local table \texttt{l\{layer\}\_all\_pcaps.parquet}; calls the
      layer's engineering function to produce a per-flow table of engineered columns;
      writes that table on its own to \texttt{l\{layer\}\_all\_fe.parquet} (retained
      for debugging and ablation); reads the per-flow global table
      \texttt{l\{layer\}\_all\_gf.parquet}; left-merges the engineered columns onto it
      on the flow key; fills any engineered column left unmatched by the merge with
      zero; and writes the result to \texttt{l\{layer\}\_all\_gf\_wfe.parquet} --- the
      global table \emph{with feature engineering}. The data flow is shown in
      Figure~\ref{fig:fe-pipeline}.

NEW:  For a given layer, the pipeline executes a fixed sequence of seven steps:
      \begin{enumerate}
        \item read the per-packet local table \texttt{l\{layer\}\_all\_pcaps.parquet};
        \item call the layer's engineering function to produce a per-flow table of
              engineered columns;
        \item write that table on its own to \texttt{l\{layer\}\_all\_fe.parquet}
              (retained for debugging and ablation);
        \item read the per-flow global table \texttt{l\{layer\}\_all\_gf.parquet};
        \item left-merge the engineered columns onto it on the flow key;
        \item fill any engineered column left unmatched by the merge with zero;
        \item write the result to \texttt{l\{layer\}\_all\_gf\_wfe.parquet} --- the
              global table \emph{with feature engineering}.
      \end{enumerate}
      The data flow is shown in Figure~\ref{fig:fe-pipeline}.
```

This fixes the count question **and** the margin overflow in one move.

---

## A8. "Good try" — the "deep-packet-inspection engine" wording (§4.6)

> **Reviewer:** marks the sentence about port heuristics + magic bytes "…supplemented by an independent deep-packet-inspection engine" with **"Good try."**

**Answer:** the gentle scepticism is because calling it a "deep-packet-inspection engine" appears to contradict the thesis's central claim that classical DPI fails on encrypted traffic — the very point you make well later (the reviewer praises §5.4.3: "Good distinction between classical DPI and what is done here"). The engine is nDPI, used for **metadata-level protocol identification**, not payload inspection. Name it that way here and forward-reference §5.4.3 so the apparent contradiction never lands.

**Suggested edit — §4.6** (`Chapters/4-feature_extraction.tex`, lines ~397–399):

```
OLD:  the application-layer signals are later cross-checked and supplemented by an
      independent deep-packet-inspection engine in Chapter~\ref{cha:feature-engineering}.

NEW:  the application-layer signals are later cross-checked and supplemented by an
      independent protocol-classification engine (nDPI) in
      Chapter~\ref{cha:feature-engineering}. As Section~\ref{sec:fe-ndpi} makes precise,
      that engine is used only for metadata-level protocol identification --- not the
      classical payload inspection that encryption defeats.
```

---

# Part B — Notation pass: percentages with two decimals (§8.2, §8.3, §8.6, §8.7, §8.8)

> **Reviewer, repeatedly:** "Please, stick to the percentual notation with 2 decimals."

The **Results** chapter already reports everything as `12.57%`, `7.82%`, etc. The **Discussion** chapter is the only place that still uses decimals (`0.126`, `0.078`, …). Converting the Discussion to match fixes the reviewer's complaint *and* removes a Results↔Discussion notation mismatch.

**Important — use the Results tables' exact values, not `decimal × 100`.** Several Discussion decimals are rounded and do **not** equal the precise table figure (e.g. `0.126` for L3 HTER, but Table 7.2 says **12.57%**, not 12.60%). The conversions below use the authoritative table values so the two chapters agree to two decimals.

### §8.2 (`Why the Layers Are Strong Together`)

| OLD | NEW |
|---|---|
| "the best single-layer model deploys at an HTER of **0.126** (L3)" | "…an HTER of **12.57\%** (L3)" |
| "The full L2+L3+L4 model deploys at **0.067 to 0.078**." | "…deploys at **6.70\% to 7.82\%**." |

### §8.3 (`The Application Layer Costs Validation EER…`)

| OLD | NEW |
|---|---|
| "L2 goes from **0.022 to 0.028** when L4 is added" | "L2 goes from **2.25\% to 2.83\%** when L4 is added" |
| "L3 goes from **0.037 to 0.067**" | "L3 goes from **3.68\% to 6.74\%**" |
| "L2+L3 goes from **0.008 to 0.028**" | "L2+L3 goes from **0.83\% to 2.83\%**" |
| "its false-negative rate is **23.5\%**" | "…is **23.50\%**" |
| "lowers the false-negative rate to **12.8\%** and the HTER from **0.121 to 0.078**" | "…to **12.75\%** and the HTER from **12.11\% to 7.82\%**" |

### §8.6 (`The Evaluation Protocol`)

| OLD | NEW |
|---|---|
| "chronological whole-capture split (HTER **0.067--0.078**)" | "…(HTER **6.70\%--7.82\%**)" |
| "a random split that still keeps each capture whole (**0.015--0.016**)" | "…(**1.50\%--1.61\%**)" |
| "a fully random split of the flows (about **0.004**, AUC above 0.99)" | "…(about **0.39\%--0.43\%**, AUC above 0.99)" |
| "the drop from roughly **0.07 to 0.015**" | "the drop from **7.82\% to 1.61\%** (mean pooling)" |

### §8.7 (`The Committee of Specialists`) — convert **and** see Part E1 before finalising

| OLD | NEW |
|---|---|
| "already reaches a deployed HTER of about **0.071**" | "…about **7.14\%**" |
| "beats the single model under mean pooling (**0.078**), though not under attention (**0.067**)" | "…(**7.82\%**), though not under attention (**6.70\%**)" |
| "a frozen committee with a combiner deploys around **0.116**, a fully fine-tuned combiner around **0.091**, and a validation-weighted ensemble of the two around **0.087**" | notation only → "…around **11.60\%**, …around **9.10\%**, …around **8.70\%**" — **but the numbers themselves need reconciling with §7.7 first; see Part E1** |
| "the warm-started committee got worse under full fine-tuning, **0.142 to 0.152**" | "…**14.20\% to 15.20\%**" |

> ⚠️ §8.7's figures are converted literally here (×100). Whether they are the *right* figures — and whether they agree with Table 7.6 — is the open question in **Part E1**. Settle E1, then apply the percentage form to the agreed numbers.

### §8.8 (`The Two Baseline Families, in Brief`)

| OLD | NEW |
|---|---|
| "validation EER **0.049** against the MLP's **0.008**" | "validation EER **4.91\%** against the MLP's **0.83\%**" |
| "deploys at about **0.29**, roughly four times the supervised error" | "deploys at about **29\%**, roughly four times the supervised error" |

### Optional, for full consistency — §8.5 (`Concept Drift…`, reviewer said "Good job", not flagged)

§8.5 uses one-decimal percentages (`0.8%`, `12.8%`, `4.8%`) and two stray decimals. To match the two-decimal tables: `0.8%→0.83%`, `12.8%→12.75%`, `4.8%→4.82%`, `2.9%→2.90%`, `1.7%→1.67%`, `10.5%→10.50%`; and "validation EER around **0.03**, test HTER **0.067 to 0.078**" → "around **3\%**, test HTER **6.70\% to 7.82\%**"; "HTER **0.067** against **0.078**" → "**6.70\%** against **7.82\%**". Low priority, but it is the same notation point.

---

# Part C — Wording, clarity, and typography fixes

### Abstract — clarify "common-set" (§1)

> **Reviewer:** "Final sentence: common-set? Understood only after reading Section 7.5. A small clarification here would help."

Gloss it at first use, so the closing sentence needs no change. `thesis.tex` line ~126:

```
OLD:  On a common evaluation set of 323
      validation and 538 test captures, assembled from several public sources, the
      cross-layer MIL detector deploys at a test HTER of 6.7\% under attention pooling

NEW:  On a common evaluation set --- the captures every model can score, so the numbers
      are directly comparable (Section~\ref{sec:res-comparability}) --- of 323
      validation and 538 test captures, assembled from several public sources, the
      cross-layer MIL detector deploys at a test HTER of 6.7\% under attention pooling
```

### Introduction — "encrypted with TLS" is imprecise (§2 / Ch.1 Context)

> **Reviewer:** "*Most traffic today is encrypted with TLS* is not correct. TLS… governs the encryption establishment rules, but the encryption can be chosen between different families. A more generic sentence is better (*Most traffic today is encrypted*)."

`1-intro.tex` line ~9:

```
OLD:  Most traffic today is encrypted with TLS, and so is most malicious traffic.
NEW:  Most traffic today is encrypted, and so is most malicious traffic.
```

### Introduction — remove "For the reader short on time" (§1.4)

> **Reviewer:** "Remove *For the reader short of time* (that has to be implicitly, not explicitly mentioned!)."

`1-intro.tex` line ~54:

```
OLD:  For the reader short on time, the contributions of this thesis are, in the
      order the chapters develop them:
NEW:  The contributions of this thesis, in the order the chapters develop them, are:
```

### Literature review — "chiefly to establish" (§2.1)

> **Reviewer:** "What does *chiefly to establish* mean?"

`2-literature-review.tex` line ~36:

```
OLD:  are drawn on chiefly to establish where consensus exists and where open
      problems remain.
NEW:  are used mainly to establish where the field agrees and where open problems
      remain.
```

### §4.3 — long line on PcapReader

> **Reviewer:** "Rephrase the sentence to avoid long line: *PcapReader [25] is used to parse packets, …*"

The overfull line is caused by `\path{PcapReader}` (unbreakable). `4-feature_extraction.tex` lines ~220–223:

```
OLD:  Packets are parsed with Scapy's \path{PcapReader}~\cite{scapy}, which is a
      streaming iterator: it yields one packet at a time and never materialises the
      whole capture in memory.
NEW:  \texttt{PcapReader}~\cite{scapy} is used to parse packets one at a time. It is a
      streaming iterator: it never materialises the whole capture in memory.
```

### §4.9 — long line on "CSV/JSON"

> **Reviewer:** "…does not fit within margins. Try to reorder it, or to substitute the CSV/JSON by CSV or JSON."

`4-feature_extraction.tex` lines ~519–520:

```
OLD:  For small datasets a verbose mode additionally emits per-packet JSON, per-capture
      CSV/JSON, and the benign/malicious split tables, which are useful for debugging
NEW:  For small datasets a verbose mode additionally emits per-packet JSON, per-capture
      CSV and JSON files, and the benign/malicious split tables, which are useful for
      debugging
```

### §5 intro — "substantial" is speculative

> **Reviewer:** "*A substantial part of the discriminative signal*… is, at this point, speculative. *A potentially substantial* is more precise."

`5-feature_engineering.tex` line ~14:

```
OLD:  A substantial part of the discriminative
      signal in network traffic does not live in any single header field,
NEW:  A potentially substantial part of the discriminative
      signal in network traffic does not live in any single header field,
```

### §6.2 — "required of" → "required for"

> **Reviewer:** "change the preposition (*required for*)."

(*"required of" is also grammatically valid, but the change is harmless and is what the reviewer asked.*) `6-framework.tex` line ~81:

```
OLD:  Two properties are required of every train/validation/test
      partition produced by the framework:
NEW:  Two properties are required for every train/validation/test
      partition produced by the framework:
```

### §6.2.4 — stray backslashes inside `\path` (the reviewer's "slashes not required for verbatim")

> **Reviewer:** "the slashes are not required for verbatim (this is missed only at the end of the first paragraph)."

Inside `\path{…}` (verbatim) the underscores must **not** be escaped — the `\_` would print a literal backslash. `6-framework.tex` line ~201:

```
OLD:  \path{src.train\_model.run\_multi\_layer\_experiment\_task}
NEW:  \path{src.train_model.run_multi_layer_experiment_task}
```

*(Worth a quick grep for any other `\path{…\_…}` in the chapter — this is the only one the reviewer saw, but the same slip may exist elsewhere.)*

### §6.4.2 — "The output is the binary cross-entropy of that logit"

> **Reviewer:** "*…the binary cross-entropy of that logit against the bag label*: something is missing. It would read better *The output at training time …*"

The model's output is a logit; the cross-entropy is the *loss* computed from it. `6-framework.tex` lines ~545–546:

```
OLD:  The output is the binary cross-entropy of that logit against the bag label,
      combined into the loss of Section~\ref{sec:fw-train-loss}.
NEW:  At training time, that logit is scored against the bag label by binary
      cross-entropy, which forms the loss of Section~\ref{sec:fw-train-loss}.
```

### §6.1 — "provenance" (optional)

> **Reviewer:** "*provenance*: most English speakers would not know what this word means. Nice."

This is a half-compliment, not a required change. If you want it more accessible, `6-framework.tex` line ~32: "indifferent to the **provenance**" → "indifferent to the **origin**". Otherwise leave it.

### §8.1 — emphasise "A note on what is comparable"

> **Reviewer:** "Emphasize the sentence *A note on what is comparable*."

`8-discussion.tex` line ~29 — make it a run-in heading:

```
OLD:  A note on what is comparable. Every number here is on the common evaluation set
NEW:  \paragraph{A note on what is comparable.} Every number here is on the common
      evaluation set
```

### §8.2 — three small fixes the reviewer requested

`8-discussion.tex`, lines ~46–59:

1. Reference Table 7.9 where complementarity is asserted:
```
OLD:  The explainability evidence (Section~\ref{sec:disc-explain}) shows why they are
      complementary: the layers do not look at the same thing.
NEW:  The per-layer attributions (Table~\ref{tab:res-shap}) already show this, and the
      explainability evidence is developed in Section~\ref{sec:disc-explain}: the layers
      do not look at the same thing.
```

2. Name the baseline families:
```
OLD:  The two baseline families lack this
      property and lose by a wide margin
NEW:  The two baseline families (XGBoost and the autoencoder) lack this
      property and lose by a wide margin
```

3. Tie "learned together" to the weak-label problem (what the baselines don't handle):
```
OLD:  The flow encoder and the detector are learned together against the detection
      objective, so the representation is shaped by the task.
NEW:  The flow encoder and the detector are learned together against the bag-level
      detection objective, so even under weak, capture-level labels the representation
      is shaped by the task --- the very thing the two baseline families cannot do (the
      autoencoder never sees a malicious label, and the untrained XGBoost encoder is
      never shaped by one at all).
```

### §8.3 — section title is unclear

> **Reviewer:** "The title *Layer Costs Validation EER and Buys Deployment* is not clear. Decreases performance and increases performance is more clear."

`8-discussion.tex` line ~63:

```
OLD:  \section{The Application Layer Costs Validation EER and Buys Deployment}
NEW:  \section{The Application Layer: Worse on Validation, Better in Deployment}
```

---

# Part D — Figure 6.2 (the only figure fix)

> **Reviewer (§6.4.2):** "Figure 6.2: *Flow N*, but *h_n*. Shouldn't both numbers (n and N) be the same?"

**Figure 6.2 is `Images/MLP_Architecture.pdf`** (the PCAP-level MIL architecture, `\label{fig:fw-mlp-arch}`). The inconsistency is **inside the image asset**, not in the LaTeX — the `.tex` caption already uses a consistent index (`$\{\mathbf{h}_i\}$`, `$\mathbf{z}$`). So this is a figure-regeneration fix, not a text edit:

- In `MLP_Architecture.pdf`, make the flow index and the embedding subscript the **same symbol** — either label the flows `Flow 1 … Flow N` and the embeddings `h_1 … h_N` (recommended), or use lowercase `n` for both. Currently it mixes "Flow N" with "h_n".
- For consistency with the prose, `N` (the bag's flow count) is the natural choice: `Flow 1, …, Flow N → h_1, …, h_N → pool → z`.

No `.tex` change is required unless you also want the caption to name the count `N` explicitly.

---

# Part E — Inconsistencies I found that the review did **not** flag

*(You asked for everything to be 100% correct and consistent, not only what the supervisor caught. These are mine, not his.)*

## E1. Results §7.7 and Discussion §8.7 do not tell the same story about advanced training — **fix before submission**

This is the important one. The two sections use the phrase "fully fine-tuned" for **different architectures and different numbers**:

- **Results §7.7 (Table 7.6)** reports the **cross-layer-expert** composite: frozen "Transformers only" = **11.61%** HTER → fully fine-tuned (validation-selected) = **13.06%** HTER. Conclusion: *fine-tuning hurts.*
- **Discussion §8.7** narrates a fully fine-tuned combiner at **~9.11%** and an ensemble at **~8.7%**, i.e. fine-tuning landing *below* the frozen committee. Conclusion: *fine-tuning helps somewhat but still loses to the fixed rule.*

A reader who goes from the table (13.06%) to the prose (9.11%) under the same name "fully fine-tuned" will see a contradiction.

**I checked `results_analysis/advanced_training.csv` — every number is real; the problem is that the two sections pair them across two different composite families:**

| arch | ft_mode | val HTER | **test HTER** | where it appears |
|---|---|---|---|---|
| `cross_layer_expert` | `transformers_only` (frozen) | 3.43% | **11.61%** | §7.7 Table 7.6, row 1 (solid match) |
| `cross_layer_expert` | `full_finetune` (best-val) | 1.42% | **13.06%** | §7.7 Table 7.6, row 2 (solid match) |
| `score_combiner` | `all_frozen` | 2.22% | **10.96%** | candidate for §8.7 "frozen…combiner" |
| `score_combiner` | `full_finetune` | 4.23% | **9.11%** | §8.7 "fully fine-tuned combiner ≈ 0.091" (solid match) |

Reading the §8.7 numbers against this table: its frozen figure (`0.116` ≈ 11.6%) is closest to a **frozen committee** (11.61% cross-layer-expert, or 10.96% score-combiner), while its fully-fine-tuned figure (`0.091`) is the **score-combiner** fine-tuned (9.11%). So §8.7 narrates "frozen → fine-tuned" as an *improvement* by taking the frozen number from a committee and the fine-tuned number from the score-combiner family. §7.7's table, by contrast, holds the family fixed (`cross_layer_expert`) and shows fine-tuning *worsening* it (11.61% → 13.06%). Both are true of their own family; presented unlabelled, they contradict.

**Recommended fix (pick one):**
- **(a) Make the names explicit in both places.** In §7.7, say the table is the *cross-layer-expert* composite; in §8.7, label the 9.11%/8.7% numbers as the *score-combiner* composite. Then "fully fine-tuned (cross-layer-expert) = 13.06%" and "fully fine-tuned (score-combiner) = 9.11%" no longer collide. *(Lowest-risk; keeps both narratives.)*
- **(b) Report one family in both.** Choose the architecture you want to feature and use its numbers in both §7.7 and §8.7.

Either way, the qualitative claim must be consistent: across both families the best composite is the **fixed-rule merge (7.14%)**, and **no learned/fine-tuned composite beats the single attention model (6.70%)** — that headline is true and worth keeping; only the intermediate numbers need to be made to correspond. **Please confirm which architecture is canonical so the two sections can be aligned.**

## E2. Minor: "order of magnitude" vs "fifteen- to twentyfold"

The abstract says the protocol "moves the deployed error by roughly an order of magnitude"; Results §7.8 and Discussion §8.6 say "roughly fifteen- to twentyfold." Both are defensible (≈10× is loose; 15–20× is the measured range), but consider making the abstract say "more than an order of magnitude" so it doesn't read as an undercount of your own result. Optional.

---

# Part F — "Good job" items (acknowledged; no action needed)

So that nothing in the review is unaccounted for, these were marked positive with no change requested. No action:

- **Overall:** "Good job."
- **Abstract:** positive except the common-set gloss (Part C).
- **Introduction §1.1–1.3, §1.5:** "Convincing, clear, concise / Well motivated / Realistic, informative, humble." §1.4 contributions praised (only the lead-in phrase removed, Part C).
- **Literature Review:** theme-wise structure endorsed ("central, and important"); §2.2–2.7 all "Good"/"Very good job" (§2.7: "Demolishing but sadly true"). Only §2.1 "chiefly" reworded (Part C).
- **Background §3.1–3.5:** all "Good job" (§3.2 "a property of the relationship between two hosts… Well explained"; §3.3 "Very clearly and cleanly explained").
- **Feature extraction §4.1, §4.2, §4.4, §4.5, §4.7, §4.8:** all "Good"/"Good job" (§4.2 flow abstraction and directionality praised). §4.3, §4.6, §4.9 handled above.
- **Feature engineering §5.1, §5.3, §5.4 (incl. §5.4.3 "Good distinction between classical DPI and what is done here"), §5.5, §5.6:** all "Good job." §5.2 and the intro line handled above.
- **Detection framework §6.1, §6.3 (incl. §6.3.5 "more important than it seems"), §6.5, §6.6, §6.7 ("good idea"), §6.8 ("Good idea to place the full artifact list in an Appendix"), §6.9:** all "Good job." §6.2, §6.4.2, Fig 6.2 handled above.
- **Results §7.1, §7.3, §7.4 ("Good idea to state it like this"), §7.5 ("Well said"), §7.6, §7.7, §7.8, §7.10 ("What could not be done, can not be shown"), §7.11, §7.12:** all "Good job." §7.2 and the §7.9 "frozen" term handled above. *(But see Part E1 on §7.7.)*
- **Discussion §8.4, §8.5, §8.9:** "Good"/"Good job." Others handled above.
- **Conclusion §9.1, §9.2:** "Good job." *(Check consistency with the A2 rewrite — see the note under A2.)*

---

# Quick action checklist

**Substantive (write/answer):** A1 XGBoost-untrained (§7.2,§8.8) · A2 selection-isn't-deployment-advice (§8.3) · A3 application-layer explanation (§8.1) · A4 "frozen"→"fixed" (7 spots) · A5 shuffle rationale (§6.2.3) · A6 presence-flag rationale (§4.6) · A7 seven-steps list (§5.2) · A8 nDPI-not-DPI (§4.6)

**Notation:** B convert Discussion decimals → % (2 dp), table-backed values.

**Wording/typography:** C — abstract common-set · intro TLS · "reader short on time" · §2.1 chiefly · §4.3 PcapReader · §4.9 CSV/JSON · §5 "potentially substantial" · §6.2 "required for" · §6.2.4 `\path` backslashes · §6.4.2 logit/loss · §6.1 provenance (opt.) · §8.1 emphasise note · §8.2 Table 7.9 + baseline names + weak-label · §8.3 title.

**Figure:** D — regenerate `MLP_Architecture.pdf` with consistent `N`.

**Consistency (mine, not the reviewer's):** E1 reconcile §7.7 ↔ §8.7 advanced-training numbers **(needs your decision on canonical architecture)** · E2 abstract "order of magnitude" (optional).
