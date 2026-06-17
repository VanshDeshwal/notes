# Thesis Edit List — exact OLD → NEW (15 June 2026)

Companion to `THESIS_REVIEW_2026-06-15.md`. Every actionable suggestion below is given as the
**exact current text** and a **drop-in replacement**, organized by file in line order. Line
numbers are current as of this review; they will shift as you apply edits, so work **bottom-to-top
within each file** (apply the highest line number first) to keep the rest valid.

Notation: `DELETE` = remove the OLD text entirely. `INSERT` = no OLD text; add NEW at the anchor.
Two edits need data I cannot generate (the dataset tables) — they are marked **[needs your numbers]**.

---

## `Chapters/1-intro.tex`

### Edit 1.1 — Shortcut 4 imbalance framing contradicts your malicious-majority dataset (line 24)

**OLD:**
```
\textbf{Shortcut 4: Accuracy on imbalanced data.} Malicious flows are rare --- in any realistic capture, the overwhelming majority of traffic is benign. On imbalanced data, accuracy is one of the least informative metrics available. A model that flags nothing achieves 99\% accuracy if 1\% of the traffic is malicious. Do Xuan et al.~\cite{APT_attack_detection_2020} report 99.02\% accuracy on an advanced persistent threat (APT) dataset and headline that number, even though their precision is only 27.73\% and F1 only 40.24\% --- meaning most of their alarms are false. The reader has to read carefully to notice.
```

**NEW:**
```
\textbf{Shortcut 4: Accuracy on imbalanced data.} Within a single capture, malicious flows are typically rare: even a compromised host emits mostly benign background traffic. On data this skewed, accuracy is one of the least informative metrics available, because it is dominated by whichever class is in the majority --- benign or malicious. A detector that simply predicts the majority class everywhere can post a headline accuracy in the high nineties while detecting nothing of value. Do Xuan et al.~\cite{APT_attack_detection_2020} report 99.02\% accuracy on an advanced persistent threat (APT) dataset and headline that number, even though their precision is only 27.73\% and F1 only 40.24\% --- meaning most of their alarms are false. The reader has to read carefully to notice. (The direction of the imbalance matters less than its presence: the dataset assembled in this thesis is, at the capture level, the opposite case --- malicious captures outnumber benign --- and accuracy is just as misleading there, which is why it is never this thesis's headline number.)
```

*Rationale: your dataset is malicious-majority at the bag level (`6-framework.tex:657`, `w⁺=N⁻/N⁺`), so the "flags nothing → 99%" example points the wrong way for your own results. The new wording keeps the critique but makes it direction-agnostic and states your dataset's actual direction.*

> No edit needed at line 92 ("including the timestamp-correction step…") — it becomes correct once you add the Chapter 4 subsection (Edit 4.1).

---

## `Chapters/2-literature-review.tex`

### Edit 2.1 — leftover `\todo` margin note (line 89)

**OLD:**
```
\subsection{Units of aggregation: packets, flows, sessions, and channels}
\label{subsec:lit-granularity}
 \todo{packets, flows, sessions, and channels difference clear?}
Beyond \emph{which} fields are used, the literature differs sharply on the
```

**NEW:**
```
\subsection{Units of aggregation: packets, flows, sessions, and channels}
\label{subsec:lit-granularity}
These four units form a nested hierarchy of increasing scope: a \emph{packet} is a single framed datagram; a \emph{flow} groups packets sharing a key (classically the five-tuple); a \emph{session} groups the flows of one logical exchange; and a \emph{channel} groups all traffic between a host pair regardless of port or protocol. Beyond \emph{which} fields are used, the literature differs sharply on the
```

*Rationale: removes the margin note (which prints, since `todonotes` is loaded) and answers the question it asked with the one-sentence clarification it requested.*

### Edit 2.2 — leftover `\todo{DPI}` (line 159)

**OLD:**
```
packet inspection \todo{DPI} (DPI). The surveys are unanimous that the widespread adoption
```

**NEW:**
```
packet inspection (DPI). The surveys are unanimous that the widespread adoption
```

*Rationale: removes the margin marker; DPI is already expanded in the intro and again here.*

---

## `Chapters/3-background.tex`

### Edit 3.1 — "flow transformer" caveat is now false after the rename (line 186)

**OLD:**
```
\paragraph{The instance encoder.}
The per-flow encoder is a \emph{multilayer perceptron}: a stack of fully
connected layers with ReLU activations and dropout, of configurable depth, that
maps a flow's feature vector to a fixed-length embedding. (A note for readers of
the accompanying code: this module is named the \emph{flow transformer} there, in
the sense of a feature transformer; it is an MLP and not a self-attention
Transformer.) The same encoder is reused as the front-end of every MIL model, so
that the families below differ only in what sits \emph{after} the pooling step.
```

**NEW:**
```
\paragraph{The instance encoder.}
The per-flow encoder is a \emph{multilayer perceptron}: a stack of fully
connected layers with ReLU activations and dropout, of configurable depth, that
maps a flow's feature vector to a fixed-length embedding. The same encoder is
reused as the front-end of every MIL model, so that the families below differ
only in what sits \emph{after} the pooling step.
```

*Rationale: the code now names this `build_instance_mlp`/`build_pcap_mlp`/`ModelMLP` (branch `BugFix`); the parenthetical excuses a name that no longer exists.*

---

## `Chapters/5-feature_engineering.tex`

### Edit 5.1 — remove the resolved `\verify` note; the TCP-state group is now computed (line 195)

**OLD:**
```
\verify{The L3 manifest (\texttt{l3\_fe\_columns.txt}) additionally lists a
TCP-state anomaly group --- pure-ACK, SYN-ACK, SYN-without-ACK, FIN, RST,
bad-checksum, SYN-RST, multi-flag, retransmission, zero-window, and
flipped-connection counts --- that \texttt{feature\allowbreak\_engineering\_l3} does not
currently compute (it merges only temporal, payload, protocol-distribution,
violation, and malformed groups). Confirm whether this group is planned,
computed elsewhere, or a stale manifest entry, and either wire it in or remove
it before the manifest is reproduced in the appendix.}
```

**NEW:** `DELETE` (the new subsection in Edit 5.2 replaces it).

*Rationale: `feature_engineering_l3` now calls `compute_l3_tcp_flag_features` and `compute_l3_tcp_connection_features` and merges both (`src/dataset_creator/feature_engineering.py`); the manifest carries the group. The `\verify` macro prints as blue text in the PDF.*

### Edit 5.2 — add a subsection describing the TCP-state anomaly group (INSERT after line 293)

`INSERT` immediately after the "Duplicate Packets (L2)" subsection (after line 293, before
`\section{External-Reference Features}`):

**NEW:**
```
\subsection{TCP-State Anomalies (L3)}
\label{sec:fe-tcpstate}

The transport layer carries a dedicated group of connection-state features that
count, per flow, packets exhibiting specific TCP control-flag and connection
patterns. Each pattern produces a per-flow boolean flag and a per-flow count:
pure-ACK segments (an ACK with no data and no other control flags), SYN-ACK and
bare-SYN-without-ACK segments (the two halves of a half-open connection), FIN and
RST segments, the illegal SYN+RST combination, multi-flag segments (more control
bits set than any legal state warrants), retransmissions, zero-window
advertisements, and flipped connections (a response observed before its
initiating request). Unlike the malformed and violation groups of
Section~\ref{sec:fe-mal-viol}, which judge a packet in isolation, these features
characterise the \emph{shape of a conversation}: an excess of half-open SYNs is a
scan signature, a high RST or retransmission rate marks unstable or rejected
connectivity, and zero-window or flipped patterns expose stalled or asymmetric
exchanges. The group is computed entirely from the extractor's own per-packet TCP
fields and so requires no external data.
```

*Rationale: gives the new 20-column group the prose it currently lacks. Note the code spells one column `retransmissio` (Edit C.1) — fix the code, not the thesis.*

### Edit 5.3 — correct the L3 manifest table (21 → 43 columns) (lines 428-448)

**OLD:**
```
    \textbf{Group} & \textbf{Cols} & \textbf{Columns} \\
    \midrule
    Temporal context     & 3  & weekday / weekend / working-hours flags \\
    Payload signatures   & 10 & per-protocol (TCP/UDP/other) file-type and blacklist hit flags and counts \\
    Protocol distribution& 4  & distinct-protocol count + TCP/UDP/other shares \\
    Protocol violations  & 2  & \texttt{has\_l3\_protocol\_violation}, \texttt{l3\_protocol\_violation\_count} \\
    Malformed packets    & 2  & \texttt{has\_malformed\_packet}, \texttt{malformed\_packet\_count} \\
    \bottomrule
  \end{tabularx}
  \caption{Transport-layer (L3) engineered features (21 columns). Payload
  signatures are external-reference features (blacklist and file-type tables);
  the remainder are internal-evidence. See the \texttt{\textbackslash verify}
  note in Section~\ref{sec:fe-pipeline} regarding the additional TCP-state group
  listed in the manifest.}
```

**NEW:**
```
    \textbf{Group} & \textbf{Cols} & \textbf{Columns} \\
    \midrule
    Temporal context     & 3  & weekday / weekend / working-hours flags \\
    Payload signatures   & 12 & per-protocol (TCP/UDP/other) file-type and blacklist hit flags and counts \\
    TCP-state anomalies  & 20 & pure-ACK, SYN-ACK, SYN-without-ACK, FIN, RST, SYN-RST, multi-flag, retransmission, zero-window, flipped-connection (flag + count each) \\
    Protocol distribution& 4  & distinct-protocol count + TCP/UDP/other shares \\
    Protocol violations  & 2  & \texttt{has\_l3\_protocol\_violation}, \texttt{l3\_protocol\_violation\_count} \\
    Malformed packets    & 2  & \texttt{has\_malformed\_packet}, \texttt{malformed\_packet\_count} \\
    \bottomrule
  \end{tabularx}
  \caption{Transport-layer (L3) engineered features (43 columns). Payload
  signatures are external-reference features (blacklist and file-type tables);
  the TCP-state, temporal, distribution, violation, and malformed groups are
  internal-evidence (Section~\ref{sec:fe-tcpstate}).}
```

*Rationale: manifest `data/feature-engineering/l3_fe_columns.txt` has 43 columns (12+3+20+4+2+2). Payload was undercounted (10→12); the 20-column TCP-state group was missing; the `\verify` cross-reference is gone.*

---

## `Chapters/6-framework.tex`

### Edit 6.1 — delete the commented-out dataset TODO block (lines 30-39)

**OLD:**
```
% TODO: commit to the dataset(s) and replace the placeholders below.  The
% expected content of this section is:
%   - dataset name(s) and citation(s);
%   - total number of PCAP files, broken down by class (benign / malicious);
%   - the time span covered by the captures;
%   - the per-class flow and packet totals (Table~\ref{tab:fw-dataset}).
% The numerical bookkeeping is performed automatically by
% lib.libdatasplitting.libds\_helper.get\_raw\_dataset\_stats() before any
% splitting takes place; the same helper produces a per-split breakdown which
% appears in Section~\ref{sec:fw-split}.
```

**NEW:** `DELETE` (it is a comment, so it does not print — but remove it once the section is written so it cannot mislead a future reader).

### Edit 6.2 — fill the dataset-composition table **[needs your numbers]** (lines 57-60)

**OLD:**
```
    Benign     & TODO  & TODO  & TODO    & TODO      \\
    Malicious  & TODO  & TODO  & TODO    & TODO      \\
    \midrule
    Total      & TODO  & TODO  & TODO    & TODO      \\
```

**NEW:** replace each `TODO` with the figure from `get_raw_dataset_stats()` run on your merged
GF parquet, e.g.:
```
    Benign     & $N_b$  & $F_b$  & $P_b$    & start--end  \\
    Malicious  & $N_m$  & $F_m$  & $P_m$    & start--end  \\
    \midrule
    Total      & $N$    & $F$    & $P$      & full span   \\
```
*I cannot generate these — the dataset parquets are not in the repo. Run `lib.libdatasplitting.libds_helper.get_raw_dataset_stats()` and paste. This table cannot ship blank.* The same applies to `tab:fw-split-report` (lines 241-243): either fill from `generate_split_report` or replace the literal `TODO` cells with clearly-marked example values, since the caption already says "illustrative".

### Edit 6.3 — "two" → "three" methodological contributions (line 75)

**OLD:**
```
The splitting strategy is the first of the framework's two methodological
contributions.  Two properties are required of every train/validation/test
```

**NEW:**
```
The splitting strategy is the first of the framework's three methodological
contributions.  Two properties are required of every train/validation/test
```

*Rationale: §6.7 calls the metric choice "the framework's third methodological contribution" (line 782). The three are: splitting, weak-label/MIL, and metrics. (Optionally, add "the second of the framework's three methodological contributions" to the opening of §6.3 so the count is explicit.)*

### Edit 6.4 — figure caption claims an exclusion the code does not perform (line 159)

**OLD:**
```
           across splits, and overlapping PCAPs that would straddle a
           boundary are excluded from training so that the boundary
           remains clean.}
```

**NEW:**
```
           across splits. Where two PCAPs overlap in time, no natural gap
           exists between them, so the boundary is simply placed at the
           nearest genuine gap instead.}
```

*Rationale: `apply_natural_gap_split` (`lib/libdatasplitting/libds.py:9-80`) excludes nothing; it only slices at gaps. The new text describes what the code actually does.*

### Edit 6.5 — stale encoder function name (line 302)

**OLD:**
```
The first ingredient -- the \emph{instance encoder} -- is a multilayer
perceptron defined by
\texttt{lib.libai.libtraining\_utils.build\_pcap\_transformer()} and reproduced
```

**NEW:**
```
The first ingredient -- the \emph{instance encoder} -- is a multilayer
perceptron defined by
\texttt{lib.libai.libtraining\_utils.build\_pcap\_mlp()} and reproduced
```

### Edit 6.6 — delete the obsolete naming caveat + broken citation (line 334)

**OLD:**
```
A naming caveat must be flagged for readers of the accompanying source code.
The module is called the \emph{flow transformer} (or
\texttt{build\_pcap\_transformer}) in the codebase, in the sense of a feature
transformer that maps a raw vector to an embedding.  It is an MLP, not a
self-attention Transformer in the sense of Vaswani et al.~\cite{TODO};
the naming is historical and refers to the role of the module in the pipeline,
not to its internal architecture.
```

**NEW:** `DELETE`.

*Rationale: removes (a) the stale "flow transformer / build_pcap_transformer" name, (b) `\cite{TODO}`, which is a broken citation — there is no Vaswani entry in `references.bib`. If you prefer to keep one clarifying sentence, use: "The encoder is a feed-forward MLP, not a self-attention Transformer." and add the Vaswani 2017 reference to `references.bib` if you cite it.*

### Edit 6.7 — (optional) acknowledge your dataset's imbalance direction in §6.7.1 (line 798)

**OLD:**
```
metric; it appears only in supporting tables.
```

**NEW:**
```
metric; it appears only in supporting tables.  The same caution applies in the
opposite direction to the dataset used here, whose captures are
malicious-majority at the bag level: a trivial detector that flagged every
capture would likewise post a high accuracy while being useless.
```

*Rationale: §6.7.1 currently argues only the benign-majority case ("predict benign → high accuracy"). Your evaluation set is malicious-majority, so the trivial baseline is the opposite. One sentence closes the gap.*

---

## `Chapters/4-feature_extraction.tex`

### Edit 4.1 — add the promised timestamp-correction subsection (INSERT after line 239)

The intro (line 92) and background (line 78) both say Chapter 4 describes the
timestamp-correction mechanics, but no such section exists. `INSERT` a subsection inside
§4.3 (after the "TCP stream reassembly (L3)" paragraph, line 239):

**NEW:**
```
\paragraph{Timestamp correction.}
The chronological split of Chapter~\ref{cha:framework} is only meaningful if every
capture carries a real wall-clock time. Some publicly distributed PCAPs instead
carry \emph{relative} or near-epoch timestamps that begin near zero. The pipeline
therefore detects any capture whose earliest packet time falls before a sanity
threshold (roughly the year 2001, i.e.\ a Unix time below $10^{9}$) and
reconstructs a real date for it: first from a date embedded in the filename
(\texttt{YYYY-MM-DD}), and failing that from a small table of known per-dataset
release dates (\texttt{DATASET\_ANCHORS}). The affected timestamps are then shifted
by the offset between the detected anchor and the relative origin. This correction
runs at extraction time, before the global features are sorted by
\texttt{first\_pkt\_timestamp} (\texttt{lib.libutils.libtimestamp.fix\_relative\_timestamps}),
and is re-applied idempotently before any split is computed, so that captures already
corrected are left unchanged. Every shift is logged. The point carried into
Chapter~\ref{cha:framework} is that ``time'' there always means corrected,
real-world time.
```

*Rationale: documents the contribution you just integrated (`src/pcap_extractor.py:385,723,1121`; re-applied at `src/train_model.py:76,213`). Adjust the prose if your `DATASET_ANCHORS`/threshold details differ.*

> After adding this, the §4.1 overview that says "three passes, the first two of which this
> chapter covers" still reads correctly; no further change needed there.

---

## `thesis.tex` (front matter housekeeping — prints in the PDF)

### Edit T.1 — remove filler text from the abstract (line 80)

**OLD:**
```
\begin{abstract}
This \texttt{abstract} environment provides a summary of the work, which may or may not be extensive.
The intention is that this should be limited to 1 page.

\lipsum[1]
\end{abstract}
```

**NEW:** replace with your real one-page abstract, e.g.:
```
\begin{abstract}
<your abstract text — the four-shortcuts framing, the layered MIL pipeline, and the
headline result, in ~250 words>
\end{abstract}
```
*Rationale: `\lipsum[1]` inserts random Latin filler into the document body. Remove the call (and your real abstract is required regardless).*

### Edit T.2 — replace placeholder acknowledgements (preface block, ~line 71)

**OLD:**
```
\begin{preface}
This is my acknowledgment to thank everyone who has kept me busy.
I would like to thank my supervisor, my advisor, and the entire jury.
```

**NEW:** your real acknowledgements (name the supervisor/advisor properly; remove the
placeholder phrasing).

### Edit T.3 — remove the stray advisor comment (line 57)

**OLD:**
```
%Irfan
```

**NEW:** `DELETE`.

### Edit T.4 — (optional) remove now-unused draft macros (lines 52-54)

**OLD:**
```
\newcommand{\rewrite}[1]{\textcolor{orange}{[REWRITE: #1]}}
\newcommand{\citationneeded}{\textcolor{purple}{[CITATION NEEDED]}}
\newcommand{\verified}[1]{\textcolor{dark_green}{#1}}
```

**NEW:** `DELETE` (none of `\rewrite`, `\citationneeded`, `\verified` is used in Chapters 1-6;
verify they are unused in 7-9 before deleting). You may also delete `\verify` (line 51) and the
`todonotes` package (line 58) once Edits 5.1, 2.1, 2.2 are applied. The `lipsum` `\IfFileExists`
guard (lines 64-66) is harmless to leave once the `\lipsum[1]` call is gone.

---

## Code fixes (not thesis text, but they propagate into the appendix manifest)

### Edit C.1 — misspelled feature-name `retransmissio` (branch `BugFix`)

In `lib/libfeature_engineering/libfe.py` and `data/feature-engineering/l3_fe_columns.txt`,
the columns are spelled:
```
has_retransmissio_packet
retransmissio_packet_count
```
**Fix to** `has_retransmission_packet` / `retransmission_packet_count` (add the final *n*) in the
computation **and** the manifest, before the manifest is frozen and reproduced in
Appendix `app:features`. Feature names are a permanent contract.

### Edit C.2 — branch hygiene

Merge `BugFix` → `main` (or pin the thesis to commit `71390d9`). The thesis describes `BugFix`
code; `main`/`origin/main` is stuck at `e717fbe` and predates all three changes.

---

## Structural changes (no single-line OLD/NEW — location + recommended action)

These were in the main review; they are larger than a line edit, so the "new text" is an
approach rather than a string:

- **Swap Chapters 2 and 3** (`thesis.tex:105-106`): change include order to `3-background`
  then `2-literature-review`, and update the cross-reference at `2-literature-review.tex:15`
  ("defined in §\ref{sec:bg-tcpip}") which currently points forward to a later chapter.
- **De-duplicate MIL** (`6-framework.tex:277-292`): replace the re-derivation of the
  bag/instance definition with one sentence — e.g. "Recall from §\ref{sec:bg-mil} the
  bag/instance mapping; here we give its formal realisation." — and keep only the math.
- **De-duplicate metrics** (`6-framework.tex:800-841` vs `3-background.tex:236-249`): same
  pattern — let §3.5 own the concept, §6.7 own the formal definitions.
- **Add XGBoost and autoencoder model subsections** to §6.5 (after line 588), or add one
  sentence at line 452 stating their construction is deferred to Chapter 7.
- **Describe `Random` and `RandomPcap` splits** in §6.2.2 (after line 172) and add them to the
  grid list (`6-framework.tex:752`); `RandomPcap` (`lib/libdatasplitting/libds.py:299`) is the
  control that isolates Shortcut 1 from Shortcut 2 and is worth a sentence of its own.
