# Exact changes applied to the thesis — Before / After

Every edit written into the source on 2026-06-27, in document order. **40 edits across 9 files.** Build re-verified after all of them: `pdflatex`, 0 errors, 0 undefined references.

Legend: each block shows the text **before** and **after**. Locations are by section + approximate current line.

---

## `thesis.tex` (Abstract)

**1. Gloss "common evaluation set" (≈ line 126)**

Before:
```
On a common evaluation set of 323
validation and 538 test captures, assembled from several public sources, the
```
After:
```
On a common evaluation set --- the captures every model can score, so the
numbers are directly comparable (Section~\ref{sec:res-comparability}) --- of 323
validation and 538 test captures, assembled from several public sources, the
```

**2. "order of magnitude" (≈ line 131)**

Before:
```
evaluation protocol alone moves the deployed error by roughly an order of
magnitude.
```
After:
```
evaluation protocol alone moves the deployed error by more than an order of
magnitude.
```

---

## `Chapters/1-intro.tex`

**3. "encrypted with TLS" → "encrypted" (≈ line 9)**

Before: `Most traffic today is encrypted with TLS, and so is most malicious traffic.`
After:  `Most traffic today is encrypted, and so is most malicious traffic.`

**4. Remove "For the reader short on time" (≈ line 54)**

Before:
```
For the reader short on time, the contributions of this thesis are, in the
order the chapters develop them:
```
After:
```
The contributions of this thesis, in the order the chapters develop them, are:
```

---

## `Chapters/2-literature-review.tex`

**5. "chiefly to establish" (§2.1, ≈ line 36)**

Before:
```
are drawn on chiefly to establish where consensus exists and where open
problems remain.
```
After:
```
are used mainly to establish where the field agrees and where open
problems remain.
```

---

## `Chapters/4-feature_extraction.tex`

**6. PcapReader overfull line (§4.3, ≈ line 220)**

Before:
```
Packets are parsed with Scapy's \path{PcapReader}~\cite{scapy}, which is a
streaming iterator: it yields one packet at a time and never materialises the
whole capture in memory.
```
After:
```
\texttt{PcapReader}~\cite{scapy} is used to parse packets one at a time. It is a
streaming iterator: it never materialises the whole capture in memory.
```

**7. "deep-packet-inspection engine" → nDPI clarification (§4.6, ≈ line 397)**

Before:
```
later cross-checked and supplemented by an independent deep-packet-inspection
engine in Chapter~\ref{cha:feature-engineering}.
```
After:
```
later cross-checked and supplemented by an independent protocol-classification
engine (nDPI) in Chapter~\ref{cha:feature-engineering}. As
Section~\ref{sec:fe-ndpi} makes precise, that engine is used only for
metadata-level protocol identification --- not the classical payload inspection
that encryption defeats.
```

**8. Presence-flag justification (§4.6, ≈ line 410)**

Before:
```
is represented more economically by a presence flag
and a packet count only, on the grounds that their mere appearance in a flow is
informative even when their fine-grained statistics are not.
```
After:
```
is represented more economically by a presence flag
and a packet count only, on the grounds that their mere appearance in a flow is
informative even when their fine-grained statistics are not. This is a deliberate
design economy rather than a result imported from the literature: these protocols
occur rarely in the corpus, so their fine-grained per-protocol statistics are
sparse and mostly zero --- dimensionality without dependable signal --- whereas
the bare fact that a flow uses one of them is occasionally diagnostic, the use of
IRC for botnet command-and-control being the classic example. The presence flag
and packet count keep that signal at negligible cost.
```

**9. CSV/JSON overfull line (§4.9, ≈ line 519)**

Before:
```
For small datasets a verbose mode additionally emits per-packet JSON, per-capture
CSV/JSON, and the benign/malicious split tables, which are useful for debugging
and inspection;
```
After:
```
For small datasets a verbose mode additionally emits per-packet JSON, per-capture
CSV and JSON files, and the benign/malicious split tables, which are useful for
debugging and inspection;
```

---

## `Chapters/5-feature_engineering.tex`

**10. "substantial" → "potentially substantial" (§5 intro, ≈ line 14)**

Before: `A substantial part of the discriminative`
After:  `A potentially substantial part of the discriminative`

**11. "The seven steps" run-on → numbered list (§5.2, ≈ line 130)**

Before:
```
For a given layer, the pipeline executes a fixed sequence. It reads the
per-packet local table \texttt{l\{layer\}\_all\_pcaps.parquet}; calls the
layer's engineering function to produce a per-flow table of engineered columns;
writes that table on its own to \texttt{l\{layer\}\_all\_fe.parquet} (retained
for debugging and ablation); reads the per-flow global table
\texttt{l\{layer\}\_all\_gf.parquet}; left-merges the engineered columns onto it
on the flow key; fills any engineered column left unmatched by the merge with
zero; and writes the result to \texttt{l\{layer\}\_all\_gf\_wfe.parquet} --- the
global table \emph{with feature engineering}. The data flow is shown in
Figure~\ref{fig:fe-pipeline}.
```
After:
```
For a given layer, the pipeline executes a fixed sequence of seven steps:
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

---

## `Chapters/6-framework.tex`

**12. "required of" → "required for" (§6.2, ≈ line 81)**

Before: `are required of every train/validation/test`
After:  `are required for every train/validation/test`

**13. Shuffle rationale strengthened (§6.2.3, ≈ line 194)**

Before:
```
whole: the chronological boundaries between train, validation, and test are
already fixed before shuffling occurs.
```
After:
```
whole: the chronological boundaries between train, validation, and test are
already fixed before shuffling occurs. It is worth being explicit about what
this does and does not affect: the set-level metrics this thesis reports (EER and
HTER, computed over the whole validation or test set) are independent of capture
order, so shuffling never changes a reported number. Its only effect is on
order-sensitive quantities --- running-average diagnostics, and any subsample
drawn from a split (such as the SHAP background of
Section~\ref{sec:res-explain}) --- which on a time-ordered set would otherwise be
biased toward the earliest captures. The chronological guarantee is carried
entirely by the split boundaries, which are fixed before the shuffle.
```

**14. Stray backslashes inside `\path` (§6.2.4, ≈ line 201)**

Before: `Concretely (\path{src.train\_model.run\_multi\_layer\_experiment\_task}):`
After:  `Concretely (\path{src.train_model.run_multi_layer_experiment_task}):`

**15. logit vs. loss (§6.4.2, ≈ line 545)**

Before:
```
The output is the binary cross-entropy of that logit against the bag label,
combined into the loss of Section~\ref{sec:fw-train-loss}.
```
After:
```
At training time, that logit is scored against the bag label by binary
cross-entropy, which forms the loss of Section~\ref{sec:fw-train-loss}.
```

---

## `Chapters/7-results.tex`

**16. Why the XGBoost encoder is untrained (§7.2, ≈ line 95)** — inserted after "…rather than trees against neural networks in general."

Added:
```
The encoder is untrained for two
reasons that compound. First, the labels are reliable only at the capture level
(the weak-label setting of Chapter~\ref{cha:framework}), so there is no per-flow
target with which to train a flow encoder directly. Second, the only available
signal is the bag label, and gradient-boosted trees are not differentiable, so
--- unlike the MLP, whose bag loss back-propagates through the pooling into the
encoder --- the XGBoost objective cannot train the representation that feeds it.
The encoder therefore stays at initialisation; boosted trees on a \emph{trained}
encoder's embeddings would be the fair trees-versus-network comparison and were
not run.
```

**17. frozen → fixed (§7.1 conventions, ≈ line 25)**

Before: `\textbf{Deployed performance is the test HTER at that frozen`
After:  `\textbf{Deployed performance is the test HTER at that fixed`

**18. frozen → fixed — two table captions (Tables 7.2 and 7.3, ≈ lines 125 & 158)**

Before (both): `FPR and FNR are at the frozen threshold.}`
After (both):  `FPR and FNR are at the fixed threshold.}`

**19. frozen → fixed (Table 7.8 caption, ≈ line 443)**

Before: `threshold frozen on validation. The benign-side error (FPR) is stable;`
After:  `threshold fixed on validation. The benign-side error (FPR) is stable;`

---

## `Chapters/8-discussion.tex`

**20. §8.1 application-layer sentence + "frozen model" → "trained model" (≈ line 18)**

Before:
```
Second, adding the application layer raises the
validation EER but lowers the deployed error, which is a sign that the
validation metric does not see what the application layer is good for
(Section~\ref{sec:disc-layers}). Third, when the frozen model is run on the test
set,
```
After:
```
Second, adding the application layer makes the
model look \emph{worse} on validation (a higher validation EER) yet perform
\emph{better} in deployment (a lower test error). It earns its place precisely on
the malicious traffic that has drifted away from the training period --- traffic
the validation set, drawn from near that period, does not contain --- so the
validation metric cannot reward it (Section~\ref{sec:disc-layers}). Third, when
the trained model is run on the test set at the threshold fixed on validation,
```

**21. Emphasise "A note on what is comparable" (≈ line 29)**

Before: `A note on what is comparable. Every number here is on the common evaluation set`
After:  `\paragraph{A note on what is comparable.} Every number here is on the common evaluation set`

**22. §8.2 percentages (≈ line 41)**

Before: `model deploys at an HTER of 0.126 (L3)` … `model deploys at 0.067 to 0.078.`
After:  `model deploys at an HTER of 12.57\% (L3)` … `model deploys at 6.70\% to 7.82\%.`

**23. §8.2 reference Table 7.9 (≈ line 46)**

Before:
```
The explainability evidence (Section~\ref{sec:disc-explain}) shows why they are
complementary: the layers do not look at the same thing.
```
After:
```
The per-layer attributions (Table~\ref{tab:res-shap}) already show this, and the
explainability evidence is developed in Section~\ref{sec:disc-explain}: the
layers do not look at the same thing.
```

**24. §8.2 name baseline families + weak-label phrasing (≈ line 56)**

Before:
```
encoder and the detector are learned together against the detection objective,
so the representation is shaped by the task. The two baseline families lack this
property and lose by a wide margin (Section~\ref{sec:disc-families}),
```
After:
```
encoder and the detector are learned together against the bag-level detection
objective, so even under weak, capture-level labels the representation is shaped
by the task --- the very thing the two baseline families (XGBoost and the
autoencoder) cannot do: the autoencoder never sees a malicious label, and the
untrained XGBoost encoder is never shaped by one at all. They lack this property
and lose by a wide margin (Section~\ref{sec:disc-families}),
```

**25. §8.3 section title (≈ line 63)**

Before: `\section{The Application Layer Costs Validation EER and Buys Deployment}`
After:  `\section{The Application Layer: Worse on Validation, Better in Deployment}`

**26. §8.3 percentages — EER series (≈ line 68)**

Before:
```
combination \emph{raises} the validation EER: L2 goes from 0.022 to 0.028 when
L4 is added, L3 goes from 0.037 to 0.067, and L2+L3 goes from 0.008 to 0.028.
```
After:
```
combination \emph{raises} the validation EER: L2 goes from 2.25\% to 2.83\% when
L4 is added, L3 goes from 3.68\% to 6.74\%, and L2+L3 goes from 0.83\% to 2.83\%.
```

**27. §8.3 percentages — FNR/HTER (≈ line 75)**

Before:
```
false-negative rate is 23.5\%. Adding L4 to make the full L2+L3+L4 model lowers
the false-negative rate to 12.8\% and the HTER from 0.121 to 0.078.
```
After:
```
false-negative rate is 23.50\%. Adding L4 to make the full L2+L3+L4 model lowers
the false-negative rate to 12.75\% and the HTER from 12.11\% to 7.82\%.
```

**28. §8.3 the model-selection "lesson" — the cheating fix (≈ line 87)**

Before:
```
The lesson is that the validation EER, used alone, can
prefer the wrong model, and that the full-stack model should be the deployment
choice even though a smaller model wins on validation. This is the same blindness
discussed in Section~\ref{sec:disc-drift}, seen from the angle of model
selection.
```
After:
```
The lesson is about the limits of the selection rule,
and it has to be stated carefully so as not to cheat. The validation EER, used
alone, prefers the smaller L2+L3 model; that the full L2+L3+L4 model in fact
deploys better is something only the test set reveals, and the test set plays no
part in selection (Section~\ref{sec:res-setup}). It would therefore be circular
to conclude ``deploy the full-stack model'': at deployment time only the
validation signal is available, and on that signal alone the application layer
looks harmful. The honest conclusion is narrower and more useful.
Validation-driven selection was enough to \emph{detect} this behaviour --- to show
that the validation EER and the deployed error can disagree, and why --- but it
does not by itself \emph{fix} it: it cannot choose the drift-robust model in
advance, because the evidence that the application layer pays off is precisely the
concept drift that the validation set, drawn from near the training period, cannot
contain. Making model selection concept-drift-aware would require a finer split
--- a second, chronologically later hold-out placed before deployment --- against
which the choice could be priced. This is the same blindness discussed in
Section~\ref{sec:disc-drift}, seen from the angle of model selection, and it is
taken up as future work in Section~\ref{sec:concl-future}.
```

**29. §8.5 percentages — FPR pair, ×2 occurrences (≈ lines 154 & 157)**

Before (both): `4.8\% to 2.9\%`
After (both):  `4.82\% to 2.90\%`

**30. §8.5 percentages — FNR (≈ line 153)**

Before: `false-negative rate is 0.8\% and the test false-negative rate is 12.8\% --- about`
After:  `false-negative rate is 0.83\% and the test false-negative rate is 12.75\% --- about`

**31. §8.5 percentages — attention FNR (≈ line 156)**

Before: `same: the false-negative rate goes from 1.7\% to 10.5\% while the false-positive`
After:  `same: the false-negative rate goes from 1.67\% to 10.50\% while the false-positive`

**32. §8.5 frozen → fixed (≈ line 162)**

Before: `of, so the model scores them lower and the frozen threshold lets more of them`
After:  `of, so the model scores them lower and the fixed threshold lets more of them`

**33. §8.5 percentages — EER/HTER (≈ line 173)**

Before: `(validation EER around 0.03, test HTER 0.067 to 0.078), and the gap is drift,`
After:  `(validation EER around 3\%, test HTER 6.70\% to 7.82\%), and the gap is drift,`

**34. §8.5 percentages + frozen → fixed (≈ line 177)**

Before:
```
better than mean (HTER 0.067 against 0.078) despite a slightly higher validation
EER: its scores are more stable under the shift, so the frozen threshold
transfers better.
```
After:
```
better than mean (HTER 6.70\% against 7.82\%) despite a slightly higher validation
EER: its scores are more stable under the shift, so the fixed threshold
transfers better.
```

**35. §8.6 percentages — three protocols (≈ line 188)**

Before:
```
this thesis's chronological whole-capture split (HTER 0.067--0.078), a random
split that still keeps each capture whole (0.015--0.016), and a fully random
split of the flows (about 0.004, AUC above 0.99).
```
After:
```
this thesis's chronological whole-capture split (HTER 6.70\%--7.82\%), a random
split that still keeps each capture whole (1.50\%--1.61\%), and a fully random
split of the flows (about 0.39\%--0.43\%, AUC above 0.99).
```

**36. §8.6 percentages — chronology drop (≈ line 195)**

Before: `so the drop from roughly 0.07` / `to 0.015 measures chronology alone: about fivefold`
After:  `so the drop from 7.82\%` / `to 1.61\% measures chronology alone: about fivefold`

**37. §8.8 XGBoost "why" + percentages (≈ line 256)**

Before:
```
never trained, so it sees a fixed random projection of the features. Its deficit
(validation EER 0.049 against the MLP's 0.008) measures a trained representation
against an untrained one, not trees against neural networks.
```
After:
```
never trained, so it sees a fixed random projection of the features. It is
untrained because the trees are not differentiable and the labels are only at the
bag level (Section~\ref{sec:res-family}): there is no per-flow label to train the
encoder, and no gradient path from the tree objective back into it. Its deficit
(validation EER 4.91\% against the MLP's 0.83\%) measures a trained representation
against an untrained one, not trees against neural networks.
```

**38. §8.8 autoencoder percentage (≈ line 265)**

Before: `validation-selected model deploys at about 0.29, roughly four times the`
After:  `validation-selected model deploys at about 29\%, roughly four times the`

---

## `Chapters/9-conclusion.tex` (notation extended here for consistency)

**39. §9.1 headline numbers (≈ line 23)**

Before:
```
at a validation-calibrated HTER of 0.067 under attention pooling (a
false-positive rate of 2.9\% at a false-negative rate of 10.5\%) and 0.078
under mean pooling,
```
After:
```
at a validation-calibrated HTER of 6.70\% under attention pooling (a
false-positive rate of 2.90\% at a false-negative rate of 10.50\%) and 7.82\%
under mean pooling,
```

**40. §9.2 layer numbers — two edits (≈ lines 44 & 48)**

Before: `deploys better than any single layer (HTER 0.067--0.078 against` / `0.126--0.149)`
After:  `deploys better than any single layer (HTER 6.70\%--7.82\% against` / `12.57\%--14.90\%)`

Before: `attains a \emph{lower} validation EER (0.008 against` / `0.028) yet deploys \emph{worse} (0.121 against 0.067--0.078), so the`
After:  `attains a \emph{lower} validation EER (0.83\% against` / `2.83\%) yet deploys \emph{worse} (12.11\% against 6.70\%--7.82\%), so the`

**41. §9.4 autoencoder (≈ line 67)**

Before: `The validation-selected benign-trained autoencoder deploys at about 0.29,`
After:  `The validation-selected benign-trained autoencoder deploys at about 29\%,`

---

## NOT changed (deliberately)

- **§8.7 + Conclusion §9.3** (advanced-training / composition numbers `0.071`, `0.091`, `0.087`, `0.116`, `0.142`, `0.152`, `0.078`, `0.067`): left in decimals — this is the **E1** inconsistency awaiting your decision on the canonical architecture.
- **§6.1 "provenance"**: left — the reviewer complimented the word.
- **Figure 6.2** (`Images/MLP_Architecture.pdf`): binary image, must be regenerated by hand so "Flow N" and the embedding subscript use the same index.
