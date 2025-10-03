
# Basics
## 🔑 1. Distributed System Basics

- A **distributed system** is a set of processes (like nodes in a network) that communicate by sending messages.
    
- No shared memory → everything is message passing.
    
- The system may be **synchronous** (rounds happen in lockstep with known delays) or **asynchronous** (unpredictable message delays).
    

👉 In networks: think of each process like a router or server. They don’t share memory, only packets.

---

## 🔑 2. Consensus Problem

This is the **heart** of the chapter.  
Consensus = all processes agree on **one common value**, even if some fail.

Three properties must hold:

1. **Agreement:** All non-faulty processes decide the same value.
2. **Validity:** The value chosen must come from one of the processes (no random garbage).
3. **Termination:** All non-faulty processes eventually decide.
    
👉 Analogy: deciding where to go for lunch in a group chat — everyone must end up at the same place, and it must be a restaurant someone suggested.

---

## 🔑 3. Failure Models

Different algorithms work depending on the failure type:

- **Crash failures:** A process just stops and never resumes (like a server crashing).
- **Byzantine failures:** A process behaves arbitrarily — lying, sending conflicting info, or acting maliciously.

👉 Crash = silent failure. Byzantine = malicious hacker node.

---

## 🔑 4. Communication Complexity

- We measure algorithms by **round complexity** (how many synchronous rounds until decision) and **message complexity** (how many messages total).
- Trade-off: fewer rounds often means more messages, and vice versa.

👉 In coding terms, this is like comparing **time complexity** and **space complexity**, but for distributed messaging.


# L3:
## 1. Randomized Byzantine Agreement (O(nlog⁡n) messages)

### The problem

- **System:** Synchronous
- **Failures:** Byzantine (nodes can lie, send different values to different processes)
- **Naive solution:** Deterministic protocols need a lot of messages (Ω(n2)\Omega(n^2)Ω(n2)) to handle all possible lies.
### The randomized trick

Instead of deterministically forcing agreement, processes use **random coin flips** to break deadlocks.

**High-level flow (simplified):**

1. Each process broadcasts its current value.
2. Everyone collects values → might see inconsistencies if Byzantine nodes lie.
3. If there’s a clear majority, follow it.
4. If tie or confusion, flip a random coin to choose a value.
5. Repeat for O(log⁡n)O(\log n)O(logn) rounds.

- Each round costs **O(n)O(n)O(n)** messages (everyone broadcasts).
- After log⁡n\log nlogn rounds, the probability of disagreement is extremely low.
- Total = O(nlog⁡n)O(n \log n)O(nlogn) messages.

👉 Analogy: Imagine 20 people voting, but a few are trolls sending mixed signals. If votes are confusing, everyone flips a coin. After a few rounds, the group almost surely converges.



```
Algorithm RandomizedAgreement(n processes, up to f Byzantine)

Each process p does:

1. v ← initial value (0 or 1)

2. For round = 1 to c·log n do:
      a. Broadcast v to all processes
      b. Receive values from all processes
      c. If there is a majority value among received:
             v ← majority value
         else:
             v ← coin_flip(0 or 1)   # break ties randomly

3. After c·log n rounds:
      Decide(v)

```
## Example — Randomized Byzantine Agreement (small run)

Setup (keep this in mind):

- 4 processes: **P1, P2, P3, P4**.
- At most **f = 1** Byzantine (we’ll make **P4** malicious).
- Initial values: P1 = **0**, P2 = **0**, P3 = **1**.
- We run **2 rounds** (≈ log⁡24\log_2 4log2​4).
- When a process sees a strict majority it adopts it; if there’s a tie it does a **coin flip**.

---

### Round 1 — broadcasts & receipts

- **What everyone _sends_** this round:
    
    - P1 sends `0` to all.
    - P2 sends `0` to all.
    - P3 sends `1` to all.
    - P4 (Byzantine) sends **1 to P1 and P2**, but **0 to P3** (tries to confuse).
    
- **What each correct process _receives_** (including its own value):
    
    - **P1** receives: {P1:0, P2:0, P3:1, P4:1} → counts: `0:2`, `1:2` → **tie** → coin flip → suppose **P1 flips 0**.
    - **P2** receives: same as P1 → tie → suppose **P2 flips 1**.
    - **P3** receives: {P1:0, P2:0, P3:1, P4:0} → counts: `0:3`, `1:1` → **majority is 0** → **P3 sets v = 0**.

After Round 1:

- P1 → 0, P2 → 1, P3 → 0 (P4 arbitrary).

---

### Round 2 — broadcasts & receipts

- **Broadcasts now**: P1:0, P2:1, P3:0, P4 (Byz) can still send arbitrary (say it tries 1 to everyone).
- **Receipts**:
    
    - **P1** sees {P1:0, P2:1, P3:0, P4:1} → `0:2`, `1:2` → tie → coin flip → suppose **0**.
    - **P2** sees {P1:0, P2:1, P3:0, P4:1} → tie → coin flip → suppose **0**.
    - **P3** sees same distribution → tie → suppose **0**.

After Round 2:

- P1, P2, P3 all have **0** → they **agree** on `0`. (P4 can remain arbitrary, but agreement among correct processes is reached.)
---

## 2. Crash-Fault Tolerant Agreement ((f+1)(f+1)(f+1) rounds)

### The problem

- **System:** Synchronous
- **Failures:** Crash-only (up to f processes may stop mid-way)
### Why rounds matter

- If a process crashes right after sending to a few nodes, its value may not have spread yet.
- Each round, values “flood” further across the network.
- After **f+1 rounds**, even the last-crashing process’s value has had enough hops to reach everyone.

**High-level flow:**

1. Round 1: All processes broadcast their initial values.
2. Round 2 to f+1: Each process rebroadcasts all values it has seen so far.
3. After f+1 rounds, all correct processes have the same set of values.
4. Decision rule: pick one deterministically (e.g., smallest ID’s value).

👉 Analogy: passing notes around the classroom. If some people leave early (crash), after f+1 rounds of passing, everyone still in the class has seen every note.

## Crash-fault Tolerant Agreement ((f+1)(f+1)(f+1) rounds)

### Setup

- **Model:** Synchronous rounds.
- **Failures:** Crash-only, up to f processes.
- **Goal:** Reach agreement on a value despite some processes stopping.

---

### Pseudocode

```
Algorithm CrashConsensus(f)

Each process p does:

1. v ← initial value

2. For round = 1 to f+1 do:
      a. Broadcast all values you know (initial + received so far)
      b. Receive values from all processes
      c. Add new values to your knowledge set

3. After f+1 rounds:
      Decide(minimum value in your set)   # or majority, or other fixed rule

```

---

### Why f+1f+1f+1 rounds?

- Each crash can “delay” the spread of one process’s value.
- Worst case: f processes crash one after another, each just after sending to a few peers.
- Their values may take f rounds to propagate fully.
- After **f+1 rounds**, every correct process has received the same set of values.
- Then they can apply the same deterministic rule (e.g., choose the smallest ID’s value) → agreement.

---

### Tiny Example

- **Processes:** P1 = 0, P2 = 1, P3 = 1.
- **Failures:** At most f=1.
- **Rounds needed:** f+1=2

**Round 1:**

- Everyone broadcasts their value.
- Suppose P2 crashes right after sending only to P1.
- P1’s knowledge = {0, 1}, P3’s knowledge = {0, 1? maybe missing P2}.

**Round 2:**

- P1 forwards all values {0,1} to everyone.
- Now P3 also learns {0,1}.

After 2 rounds, both P1 and P3 know {0,1} → they apply the same deterministic rule (say, pick the minimum = 0). Agreement reached.

## 📘 Randomized Byzantine Agreement (O(nlog⁡n) messages)

---

### 1. Setup

- **Model:** synchronous, message passing.
- **Failures:** up to f<n/3f < n/3f<n/3 Byzantine.
- **Goal:** consensus (agreement, validity, termination).

Each process maintains a local value vvv. In each round, processes exchange values and update according to rules (majority or coin flip).

---

### 2. The Algorithm (formal)

At each process ppp:

1. Start with input vpv_pvp​.
2. For round r=1,2,…,clog⁡nr = 1, 2, \ldots, c \log nr=1,2,…,clogn:
    - Broadcast current value vpv_pvp​.
    - Collect values from all processes.
    - If at least (n−f)(n - f)(n−f) messages agree on some value bbb:
        - Set vp:=bv_p := bvp​:=b.
    - Else if no strong majority:
        - Set vp:=v_p :=vp​:= coin_flip(0,1).
3. After clog⁡nc \log nclogn rounds, **decide vpv_pvp​**.

---

## 3. Lemmas and Proofs

### **Lemma 1 (Validity).**

If all correct processes start with the same value b, then they always decide b.

**Proof idea:**

- If everyone starts with bbb, then all broadcasts carry bbb.
- Even if Byzantine processes send arbitrary messages, at least n−fn-fn−f identical messages exist for bbb.
- Since n>3f, we know n−f>2fn - f > 2fn−f>2f. So the correct value bbb dominates.
- Therefore, no coin flips occur and bbb is kept forever.
    

👉 Example: n=7,f=2n=7, f=2n=7,f=2. Suppose all correct start with 0. Then at least 5 processes send 0, while at most 2 can send lies. Every correct process sees ≥5 zeros → majority is clear → they all keep 0.

---

### **Lemma 2 (Agreement Progress).**

If in some round all correct processes use the same value bbb, then from that round onward they stick to bbb.

**Proof idea:**

- Suppose all correct use bbb.
    
- Then at least n−fn-fn−f identical messages with value bbb exist.
    
- Any correct process receives ≥ n−fn-fn−f consistent messages → it updates to bbb.
    
- No process ever flips away.
    

👉 Once unanimity happens, it is **stable forever**.

---

### **Lemma 3 (Randomization reduces disagreement).**

Suppose in a round there is no clear majority, so processes flip coins. Then with probability ≥ 1/21/21/2, all correct processes choose the same bit.

**Proof sketch:**

- Each correct process flips an **independent unbiased coin**.
    
- With probability 1/21/21/2, they all flip 0; with probability 1/21/21/2, they all flip 1.
    
- Byzantine processes cannot influence these flips.
    
- Therefore, chance of convergence in this round is ≥ 1/21/21/2.
    

👉 Example: P1, P2, P3 correct; P4 Byzantine. If P1 flips 0, P2 flips 0, P3 flips 0 → convergence. If not, another round reduces disagreement again.

---

### **Lemma 4 (Exponential convergence).**

Let event ErE_rEr​ = “all correct processes have the same value by end of round rrr.”  
Then:

Pr⁡[not Er]≤(12)r\Pr[\text{not }E_r] \leq \left(\tfrac{1}{2}\right)^rPr[not Er​]≤(21​)r

**Proof:**

- Each round, if disagreement persists, there is ≥ 1/2 chance it resolves (by coin flips aligning).
    
- So after rrr rounds, probability of still disagreeing ≤ (1/2)r(1/2)^r(1/2)r.
    

---

### **Theorem (Correctness).**

The algorithm achieves consensus with probability ≥ 1−1/nc1 - 1/n^c1−1/nc using O(nlog⁡n)O(n \log n)O(nlogn) messages.

**Proof sketch:**

- **Validity:** Lemma 1.
    
- **Agreement:** Lemma 2 (stability once reached) + Lemma 3 (random flips guarantee convergence with high prob.).
    
- **Termination:** After clog⁡nc \log nclogn rounds, disagreement probability ≤ (1/2)clog⁡n=1/nc(1/2)^{c \log n} = 1/n^c(1/2)clogn=1/nc.
    
- Each round uses O(n)O(n)O(n) messages (one broadcast per process).
    
- Total complexity: O(nlog⁡n)O(n \log n)O(nlogn).
    

---

## 4. Worked Micro Example (again, formal lens)

- n=4,f=1n=4, f=1n=4,f=1.
    
- Initial values: P1=0, P2=0, P3=1, P4 (Byzantine).
    

**Round 1:**

- Correct processes see split → tie → coin flips.
    
- Suppose P1 flips 0, P2 flips 1, P3 flips 0.
    

**Round 2:**

- Values: P1=0, P2=1, P3=0.
    
- Still tie, flip again. Suppose all flip 0.
    
- Now unanimity achieved → stable.
    

Decision = 0.


# 🌱 First Pass: Simple Intuition (no heavy math)

### Setup

- **System:** synchronous (time moves in rounds).
    
- **Faults:** crash-only, up to fff processes can suddenly stop sending.
    
- **Goal:** all correct processes decide on the same value.
    

---

### Idea of the algorithm

- Each process starts with its own value (say 0 or 1).
    
- Every round, each process broadcasts all the values it has seen so far.
    
- If some process crashes, its value may not reach everyone immediately — but other processes can “carry it forward” in later rounds.
    
- After f+1f+1f+1 rounds, every correct process will have collected the **same set of values**.
    
- Finally, they apply a deterministic rule (e.g., choose the minimum, or majority) → consensus.
    

---

### Why f+1f+1f+1 rounds?

- Think worst case: processes crash **sequentially**.
    
- Example: In round 1, P1 crashes after sending to only 1 neighbor. That neighbor forwards P1’s value in round 2. Then maybe another process crashes, etc.
    
- Each crash delays the spread of some value by 1 round.
    
- If there are fff crashes, it may take **fff rounds** for that last hidden value to reach all.
    
- So we need **one extra round** to guarantee everyone is synchronized.
    

👉 Analogy: Think of passing notes in class. If at most fff students leave early, you need f+1f+1f+1 passes to make sure every note reaches everyone still in the room.

---

### Micro Example

- n=4,f=1n=4, f=1n=4,f=1.
    
- Initial values: P1=0, P2=1, P3=1, P4=0.
    

**Round 1:**

- Everyone broadcasts. Suppose P2 crashes after sending only to P1.
    
- Now P1 knows {0,1}, but P3 and P4 know only {0,0,1?} (missing P2).
    

**Round 2:**

- P1 rebroadcasts all values {0,1}.
    
- Now P3 and P4 also see P2’s value.
    
- Everyone has the same set {0,1}.
    
- Rule: pick minimum = 0. Consensus reached.
    

---

✅ So informally:

- The algorithm is just “keep flooding values for f+1f+1f+1 rounds.”
    
- After that, everyone has the same information, so a deterministic choice guarantees agreement.
    

---

## 📘 Second Pass: Formal Treatment (with math & lemmas)

---

### The Algorithm (formal)

At each process ppp:

1. Set Sp←{vp}S_p \gets \{v_p\}Sp​←{vp​} (initial value).
    
2. For round r=1r = 1r=1 to f+1f+1f+1:
    
    - Send SpS_pSp​ to all processes.
        
    - Receive sets SqS_qSq​ from all others.
        
    - Update Sp←Sp∪⋃SqS_p \gets S_p \cup \bigcup S_qSp​←Sp​∪⋃Sq​.
        
3. After round f+1f+1f+1, decide rule(Sp)\text{rule}(S_p)rule(Sp​) (e.g., min element).
    

---

### Lemma 1 (Validity).

If all processes start with the same value vvv, then they all decide vvv.

**Proof:**

- In round 1, every process broadcasts vvv.
    
- Each correct process receives only vvv.
    
- So Sp={v}S_p = \{v\}Sp​={v} always.
    
- Decision rule outputs vvv. ✅
    

---

### Lemma 2 (Set growth is monotone).

The sets SpS_pSp​ are non-decreasing: once a value enters some SpS_pSp​, it is never lost.

**Proof:**

- Update rule is union: Sp←Sp∪SqS_p \gets S_p \cup S_qSp​←Sp​∪Sq​.
    
- Unions never delete elements. ✅
    

---

### Lemma 3 (Propagation under crashes).

If a value vvv is known by some correct process in round rrr, then by round r+1r+1r+1, at least one more correct process learns vvv, unless all correct already know it.

**Proof idea:**

- The process that knows vvv sends it to all in round r+1r+1r+1.
    
- Unless it crashes before sending to anyone, at least some correct neighbor receives vvv.
    
- Repetition ensures vvv spreads.
    

---

### Lemma 4 (Bound on rounds).

After f+1f+1f+1 rounds, every value known by any correct process is known by all correct processes.

**Proof:**

- Each crash can delay a value by at most 1 round.
    
- In the worst case, fff processes crash sequentially, hiding a value for fff rounds.
    
- But by round f+1f+1f+1, all crashes are exhausted.
    
- Thus, all values have propagated to everyone. ✅
    

---

### Theorem (Correctness of Crash-fault Consensus).

The algorithm achieves consensus in f+1f+1f+1 rounds.

**Proof:**

- **Validity:** Lemma 1.
    
- **Agreement:** After f+1f+1f+1 rounds, all correct processes have identical sets (Lemma 4). Applying the same decision rule → same output.
    
- **Termination:** Always halts after f+1f+1f+1 rounds.
    

---

### Complexity

- Each round: each process broadcasts once → O(n)O(n)O(n) messages per round.
    
- Total = O(nf)O(nf)O(nf) messages, f+1f+1f+1 rounds.

# L4:

## 📘 Fundamental Resilience Bound for Byzantine Agreement

---

## 1. Statement of the Bound

**Byzantine Agreement is solvable in a synchronous message-passing system _iff_

n>3f

where n = number of processes, f = maximum number of Byzantine faults.**

- If n≤3fn \leq 3fn≤3f, Byzantine agreement is **impossible**.
    
- If n>3fn > 3fn>3f, it is **possible** (algorithms exist, e.g., OM(f), King algorithm).
    

---

## 2. Intuition (story form)

Imagine 3 generals (n=3n=3n=3) and 1 traitor (f=1f=1f=1):

- The traitor can tell **General A** “attack” and **General B** “retreat.”
    
- Since A and B only have each other + the traitor, they can’t distinguish whether:
    
    - The traitor is lying, or
        
    - The other general is the traitor.
        
- Result: disagreement is inevitable.
    

👉 More generally: if n≤3fn \leq 3fn≤3f, the faulty processes can **split the correct processes into 3 groups** and feed each group conflicting values.

---

## 3. Formal Proof Sketch (as in the study material)

### Case 1: n≤3fn \leq 3fn≤3f. (Impossibility)

- Partition processes into 3 sets: A,B,CA, B, CA,B,C.
    
    - A,BA, BA,B are groups of correct processes.
        
    - CCC is all faulty processes.
        
- Faulty processes in CCC send value 0 to all in AAA, and value 1 to all in BBB.
    
- Now, from AAA’s perspective:
    
    - All correct peers + some faulty are saying “0.”
        
- From BBB’s perspective:
    
    - All correct peers + some faulty are saying “1.”
        
- Neither group can distinguish whether the **other group is faulty** or not.
    
- Therefore, AAA must decide 0, BBB must decide 1 → **agreement is broken**.
    

---

### Case 2: n>3fn > 3fn>3f. (Possibility)

- There exist deterministic algorithms (e.g., OM(f), King algorithm) that solve Byzantine Agreement.
    
- Key reason: with n>3fn > 3fn>3f, the set of correct processes is larger than any two disjoint sets of faulty processes, so **a correct majority can always be extracted**.
    

---

## 4. Key Lemma Used in Proof

If n>3fn > 3fn>3f, then any two sets of size n−fn-fn−f must intersect in at least f+1f+1f+1 processes.

**Proof:**

∣(n−f)+(n−f)−n∣=2n−2f−n=n−2f| (n-f) + (n-f) - n | = 2n - 2f - n = n - 2f∣(n−f)+(n−f)−n∣=2n−2f−n=n−2f

Since n>3f  ⟹  n−2f>fn > 3f \implies n - 2f > fn>3f⟹n−2f>f, so intersection ≥ f+1f+1f+1.

👉 Meaning: whenever two groups of processes collect messages from n−fn-fn−f senders, there is guaranteed overlap of at least one **correct process** (since f+1f+1f+1 > number of faulty processes). This ensures consistency.


# 1️⃣ Oral Messages Algorithm (OM(f)) — _Exponential Algorithm_

### Idea

- Proposed by **Lamport, Shostak, Pease (1982)** in the _Byzantine Generals Problem_.
    
- Works in synchronous systems, tolerates up to fff Byzantine processes, provided **n>3fn > 3fn>3f**.
    
- Runs in **f+1f+1f+1 rounds**, but message complexity is **exponential in fff** (O(nf+1)O(n^{f+1})O(nf+1)).
    

### How it works

- One process is the **commander**; others are **lieutenants**.
    
- Goal: lieutenants must agree on the commander’s order (attack/retreat).
    
- **Recursive message passing**:
    
    - OM(0): Commander sends value to all; each lieutenant decides what it received.
        
    - OM(f): Commander sends value; then each lieutenant acts as a commander and forwards what it got to others, using OM(f-1).
        
- Decision rule: lieutenants take **majority** of the received values.
    

### Properties

- Correct if n>3fn > 3fn>3f.
    
- Guarantees **agreement, validity, termination**.
    
- **Downside:** message explosion → exponential in fff.
    

👉 In exams: usually mentioned as a baseline, not for detailed steps.

---

# 2️⃣ King Algorithm (King-Phase Algorithm)

This is the **efficient deterministic BA algorithm** discussed in detail in Attiya & Welch.

---

## 2.1. Intuition

- The algorithm proceeds in **phases** (each with 2 rounds).
    
- Each phase has a designated **king process**.
    
- The king’s role: **break ties** when processes are confused.
    
- Since there are f+1f+1f+1 phases and at most fff Byzantine processes, at least one king is guaranteed to be correct.
    
- When a correct king acts, all correct processes converge to the same value.
    

---

## 2.2. Algorithm Outline

Let vpv_pvp​ = current value of process ppp.

For **f+1f+1f+1 phases**:

1. **Round 1 (Broadcast values):**
    
    - Each process broadcasts its current value.
        
    - Each process collects values.
        
    - If some value appears in at least n−fn-fn−f messages → adopt it.
        
2. **Round 2 (King’s rule):**
    
    - The king for this phase broadcasts its value.
        
    - If no strong majority was seen in Round 1, processes adopt the king’s value.
        

After f+1f+1f+1 phases, decide on current value.

---

## 2.3. Why it works

- If a value reaches n−fn-fn−f support, it dominates (since ≥ 2f+12f+12f+1 > half the processes are correct).
    
- If no such majority, Byzantine nodes caused a tie → king resolves it.
    
- With f+1f+1f+1 kings, at least 1 king is correct.
    
- In the phase with a correct king, all correct processes adopt the same value → from then on, stability holds.
    

---

## 2.4. Lemmas and Proofs

### Lemma 1 (Stability of strong majority)

If a value vvv is supported by ≥ n−fn-fn−f processes in some round, then in all later rounds all correct processes will keep vvv.

**Proof:**

- n−fn-fn−f means at least n−2fn-2fn−2f correct processes support vvv.
    
- Since n>3f  ⟹  n−2f>fn > 3f \implies n-2f > fn>3f⟹n−2f>f, at least one correct process always broadcasts vvv.
    
- So vvv never disappears. ✅
    

---

### Lemma 2 (King correctness guarantees agreement)

If the king is correct in a phase, then all correct processes adopt the same value by end of that phase.

**Proof:**

- Suppose no strong majority is seen.
    
- Then all correct processes rely on king’s broadcast.
    
- Since king is correct, all receive the same value.
    
- So agreement is restored. ✅
    

---

### Theorem (Correctness of King Algorithm)

With n>3fn > 3fn>3f, the King Algorithm achieves Byzantine Agreement in f+1f+1f+1 phases.

**Proof:**

- **Validity:** If all start with same value, it never changes (Lemma 1).
    
- **Agreement:** Eventually a correct king acts (since f+1f+1f+1 kings > fff faulty). By Lemma 2, agreement is achieved then maintained.
    
- **Termination:** Algorithm halts after f+1f+1f+1 phases. ✅
    

---

## 2.5. Complexity

- Each phase has 2 rounds, each round = 1 broadcast per process.
    
- **Total rounds:** 2(f+1)2(f+1)2(f+1).
    
- **Messages:** O(nf)O(nf)O(nf), polynomial (much better than OM(f)).
    

---

## 3. Summary

|Algorithm|Rounds|Messages|Works if|Notes|
|---|---|---|---|---|
|OM(f) (Exponential)|f+1f+1f+1|O(nf+1)O(n^{f+1})O(nf+1)|n>3fn > 3fn>3f|Historical baseline|
|King Algorithm|2(f+1)2(f+1)2(f+1)|O(nf)O(nf)O(nf)|n>3fn > 3fn>3f|Efficient, practical|

# 1) OM(f) — Oral Messages Algorithm (brief recursive trace)

**Setup**

- n=4n=4n=4 processes: Commander = **C**, Lieutenants **L1, L2, L3**.
    
- f=1f=1f=1 (so OM(1) is used; recursion depth = 1).
    
- Commander’s true order = **ATTACK (1)**.
    
- Assume **L3 is Byzantine** (it may lie when forwarding).
    
- Goal: all _non-faulty_ lieutenants agree on the commander’s order.
    

**Algorithm idea (OM(1))**

- Round 0: Commander sends its value to every lieutenant.
    
- Then, for OM(1): each lieutenant who received a value acts as “commander” and sends that value to every _other_ lieutenant (this is the recursive step OM(0)).
    
- Finally, each lieutenant takes the majority of the values it collected (including the direct commander value and those forwarded by others).
    

**Run (step-by-step)**

**Step A — Commander → Lieutenants (OM(1) first send)**

- C → L1: `1`
    
- C → L2: `1`
    
- C → L3: `1`
    

(so all lieutenants got `1` from C)

**Step B — Each lieutenant forwards what it got to the other lieutenants (OM(0) step)**

- L1 forwards to L2, L3: sends `1`.
    
- L2 forwards to L1, L3: sends `1`.
    
- L3 is Byzantine: it may forward incorrect values. Suppose L3 forwards:
    
    - To L1: `0` (lie)
        
    - To L2: `1` (truth)
        

**Step C — What each lieutenant _receives_ and decides (majority)**

- **L1** receives:
    
    - from C: `1` (its own copy)
        
    - from L2: `1`
        
    - from L3: `0` → values = {1,1,0} → majority = `1`.
        
- **L2** receives:
    
    - from C: `1`
        
    - from L1: `1`
        
    - from L3: `1` → values = {1,1,1} → majority = `1`.
        
- **L3** (Byzantine) may do anything; we ignore its decision.
    

**Outcome**

- L1 and L2 both decide `1` (ATTACK) — agreement among correct lieutenants. OM(1) succeeded.
    

**Notes**

- If the commander had sent `0` but a lying lieutenant attempted to confuse others by sending `1` to some, the recursive majority would still filter out the single liar because the majority among correct messengers wins.
    
- Message count quickly grows with larger fff because at level iii every process forwards to all others, leading to O(nf+1)O(n^{f+1})O(nf+1) messages.
    

---

# 2) King-Phase Algorithm — step-by-step example

**Setup**

- n=4n=4n=4 processes: **P1, P2, P3, P4**.
    
- f=1f=1f=1. So algorithm runs for f+1=2f+1=2f+1=2 phases. (Each phase has 2 rounds in many variants: a value-exchange round and king round.)
    
- Initial values:
    
    - P1: `0`
        
    - P2: `1`
        
    - P3: `1`
        
    - P4: Byzantine (we’ll specify behavior)
        
- King order: Phase 1 king = **P1**, Phase 2 king = **P2** (just an example rotation).
    

**High-level per phase**

1. **Round A (v-exchange):** everyone broadcasts current value; if some value has ≥ n−f=3n-f = 3n−f=3 supports, adopt it (strong majority).
    
2. **Round B (king):** king broadcasts its value. If a process saw no strong majority in Round A, it adopts king’s value. Otherwise it keeps the majority value it had.
    

We’ll show two scenarios to illustrate the crucial point: (a) _faulty king_, (b) _correct king_.

---

### Scenario 1 — Phase 1: faulty king (king = P1 but P1 is actually not faulty here; to show faulty king case we can make P4 the king instead; let’s do that)

Reset: let kings be Phase1=P4 (Byzantine), Phase2=P2 (correct). Initial values same.

#### Phase 1 — Round A (everyone broadcasts initial values)

- P1 sends `0`
    
- P2 sends `1`
    
- P3 sends `1`
    
- P4 (Byzantine) can send inconsistent things; suppose it sends:
    
    - to P1: `0`
        
    - to P2: `1`
        
    - to P3: `1`
        

What each non-faulty process _sees_:

- P1 sees: {P1:0, P2:1, P3:1, P4:0} → counts `0:2`, `1:2` → no strong majority (needs 3).
    
- P2 sees: {0,1,1,1} → counts `1:3` → **strong majority 1**.
    
- P3 sees: {0,1,1,1} → counts `1:3` → **strong majority 1**.
    

So P2 and P3 adopt `1` (and will keep it if a strong majority is seen); P1 has no strong majority.

#### Phase 1 — Round B (king = P4 broadcasts)

- P4 is Byzantine; suppose it tells everyone `0` (intends to confuse).
    
- Now processes update:
    
    - P1 had no strong majority → adopts king’s value `0`.
        
    - P2 and P3 already had strong majority `1` → they **keep 1**.
        

End of Phase 1 values:

- P1 → `0`
    
- P2 → `1`
    
- P3 → `1`
    

Agreement not yet achieved.

---

#### Phase 2 — Round A (broadcast current values)

Now kings: Phase 2 king = P2 (correct). Broadcasts:

- P1 sends `0`
    
- P2 sends `1`
    
- P3 sends `1`
    
- P4 (Byzantine) may send inconsistent values; suppose it tries to keep confusion: to P1: `0`, to P2: `1`, to P3: `1`.
    

What each sees:

- P1 sees {0,1,1,0} → `0:2`, `1:2` → no strong majority.
    
- P2 sees {0,1,1,1} → `1:3` → strong majority `1`.
    
- P3 sees {0,1,1,1} → `1:3` → strong majority `1`.
    

So P2 and P3 keep `1`, P1 still no majority.

#### Phase 2 — Round B (king = P2 broadcasts; P2 is correct and broadcasts `1`)

- P2, being correct, broadcasts `1`. Everyone receives the same `1` from the king.
    
- Processes that had no strong majority (P1) adopt king’s `1`.
    
- Those with strong majority (P2,P3) keep `1`.
    

End of Phase 2 values:

- P1 → `1`
    
- P2 → `1`
    
- P3 → `1`
    

All correct processes now agree on `1`. Since P2 was a correct king in phase 2, it pulled everyone to the same value. Agreement is stable after that.

**Summary of this run**

- Phase 1 had a _faulty king_ so it didn’t resolve global agreement.
    
- Phase 2 had a _correct king_; that phase enforced a unanimous value.
    
- By design we do f+1=2f+1=2f+1=2 phases; at least one king is correct → guarantees eventual agreement.
    

---

## Key comparisons by example

- **OM(f)**: recursive forwarding + local majorities — robust but message-explosive as fff grows. In our OM(1) trace, a single lying lieutenant couldn’t mislead the honest majority because messages were echoed and majority taken.
    
- **King algorithm**: operates in phases; kings break ties when a strong majority is absent. With ≤fff faulty processes and f+1f+1f+1 phases, at least one phase has a correct king and that phase forces all honest processes to the same value.

# L5:
## 1️⃣ Fundamental Setup

- We have nnn processes, up to fff are Byzantine.
    
- **Resilience bound still applies**: solvable only if n>3fn > 3fn>3f.
    
- Randomized algorithms aim to reduce rounds → sometimes **constant expected rounds (O(1))**, unlike deterministic f+1f+1f+1.
    

Both Rabin’s and Arora’s versions assume a **global random coin**:

- Every process sees the same coin toss outcome each round.
    
- Byzantine adversary knows the system state but cannot predict future coin flips.
    

---

## 2️⃣ Rabin’s Randomized BA Protocol (Las Vegas, O(1) rounds)

### Algorithm (each process i)

- Maintain **vote** (initially input bit bib_ibi​).
    
- Each round:
    
    1. Broadcast current vote to all.
        
    2. Collect votes (possibly inconsistent because of faulty nodes).
        
    3. Compute:
        
        - **maj** = majority value in received votes.
            
        - **tally** = count of maj.
            
    4. Rules:
        
        - If tally≥2t+1tally \geq 2t+1tally≥2t+1 (supermajority): set vote = maj.
            
        - Else: use global coin toss → Heads = 1, Tails = 0.
            
- Repeat until all good processes agree.
    

### Why it works

- If all good processes start the same, they lock in immediately.
    
- Otherwise:
    
    - With probability ≥ 1/2, coin toss aligns everyone → convergence.
        
    - Once convergence happens, stability is permanent (votes never diverge again).
        
- **Expected rounds = O(1).**
    
- **Las Vegas:** always correct, but runtime is random (constant expected).
    

---

## 3️⃣ Extended Protocol (Sanjeev Arora’s Notes)

Arora expands Rabin’s idea to more general Byzantine settings.

### Model assumptions

- Global coin.
    
- Tolerates up to n/3n/3n/3 Byzantine nodes (the max possible).
    
- The protocol uses slightly different thresholds than Rabin’s original:
    
    - **Thresholds = 5n/8 + 1, 6n/8 + 1**, depending on coin outcome.
        
    - If tally ≥ threshold → adopt maj.
        
    - Else vote = 0.
        
    - If tally ≥ 7n/8 → **decide** maj and stop.
        

### Key Lemmas

1. If all good nodes share same vote in some round → decision in O(1) rounds.
    
2. If two good nodes disagree on maj in some round → next round they all reset to 0 → convergence.
    
3. If round is “un-foiled” (no disagreement), then by the next round everyone decides.
    
4. Probability of reaching un-foiled round is constant (expected 2 rounds).
    

**Theorem:** With probability 1, agreement is reached in O(1) expected rounds (Las Vegas guarantee).

---

## 4️⃣ Intuition with Example (small case)

Say n=7n=7n=7, f=2f=2f=2.

- Good nodes: 5, Bad nodes: 2.
    
- Round 1: good nodes see different majorities because Byzantine nodes lie.
    
- Threshold condition not satisfied → all flip global coin.
    
- Suppose coin = Heads → everyone sets vote=1.
    
- Now all good nodes aligned.
    
- Next round, tally ≥ 2t+1 (i.e., 5 out of 7) → lock in.
    

---

## 5️⃣ Comparison

|Algorithm|Assumption|Byzantine bound|Rounds|Type|
|---|---|---|---|---|
|**Rabin’s BA**|Global coin, n=3t+1n=3t+1n=3t+1|f≤n/3f \leq n/3f≤n/3|O(1) expected|Las Vegas|
|**Arora’s extended version**|Global coin|f<n/3f < n/3f<n/3|O(1) expected|Las Vegas, stronger thresholds|



## 🔹 Notations

- nnn = total number of processes (nodes).
    
- fff = maximum number of Byzantine faulty processes (sometimes denoted ttt in Rabin’s paper).
    
- Correctness requires the **resilience bound**:
    
    n>3f(equivalently, f<n/3).n > 3f \quad \text{(equivalently, } f < n/3 \text{).}n>3f(equivalently, f<n/3).

So:

- Rabin’s original: uses ttt for Byzantine faults.
    
- Arora’s notes: often use fff or just assume “up to n/3n/3n/3” bad processes.
    
- In your exam answers: just be consistent — “Let fff be the number of Byzantine faults, nnn processes total, with n>3fn > 3fn>3f.”
    

---

## 🔹 Rabin’s Randomized BA Protocol (Las Vegas, O(1) expected rounds)

## Algorithm recap

Each round:

1. Every process broadcasts its current vote (0 or 1).
    
2. Everyone collects votes.
    
3. Let `maj` = majority value, `tally` = number of votes for maj.
    
4. Rule:
    
    - If `tally ≥ 2f+1` → adopt maj.
        
    - Else → adopt global random coin value.
        

Repeat until all decide.

---

## Proof Lemmas

### Lemma 1 (Stability of supermajority)

If a value gets at least 2f+12f+12f+1 votes in a round, then all correct processes adopt it and it **cannot be overturned**.

- Why: at most fff are faulty.
    
- So at least 2f+1−f=f+12f+1 - f = f+12f+1−f=f+1 correct processes support maj.
    
- Every round they keep broadcasting maj.
    
- This ensures no other value can reach supermajority.
    

---

### Lemma 2 (Coin alignment probability)

If no value gets supermajority in a round, then all correct processes adopt the **same coin value** with probability 1.

- Why: the global coin is the same for everyone.
    
- So either all flip to 0 or all to 1.
    
- With probability 1/2 they converge to 0, probability 1/2 to 1.
    

---

### Lemma 3 (Expected convergence)

Each round has probability ≥ 1/2 of achieving consensus (via Lemma 2).

- Expected number of rounds until convergence = geometric distribution with p=1/2p=1/2p=1/2.
    
- E[rounds]=1/p=2\mathbb{E}[\text{rounds}] = 1/p = 2E[rounds]=1/p=2.
    
- So **expected O(1)O(1)O(1) rounds**.
    

---

### Theorem (Correctness)

Rabin’s protocol ensures:

- **Agreement**: Once a supermajority forms or coin aligns everyone, they never diverge again.
    
- **Validity**: If all start with same value, it has supermajority from round 1 → preserved.
    
- **Termination**: Expected constant rounds (Las Vegas — always correct, runtime random).
    

---

## 🔹 Arora’s Extended Version (with thresholds)

## Algorithm recap

Uses stricter thresholds to push fault-tolerance to the maximum f<n/3f < n/3f<n/3.

Each round:

1. Broadcast votes, compute `maj`, tally.
    
2. If `tally ≥ 7n/8` → **decide maj**.
    
3. Else if `tally ≥ 5n/8` and coin = Heads → adopt maj.
    
4. Else if `tally ≥ 6n/8` and coin = Tails → adopt maj.
    
5. Else adopt 0.
    

---

## Proof Lemmas (Arora)

### Lemma A (Decision stability)

If some value reaches ≥7n/8\geq 7n/8≥7n/8, then at least (7n/8−f)(7n/8 - f)(7n/8−f) correct processes support it.  
Since f<n/3f < n/3f<n/3, this is > 2n/32n/32n/3.  
Thus no other value can ever gain enough support. Decision is safe. ✅

---

### Lemma B (Reset to 0 on disagreement)

If two correct processes see different majorities, then in the next round **all reset to 0**.

- Why: the thresholds are chosen so that Byzantine nodes can’t maintain two competing majorities.
    
- So disagreement → forced reset → system re-synchronizes.
    

---

### Lemma C (Un-foiled round leads to decision)

A round is **un-foiled** if all correct processes see the same majority.

- Then with the coin toss thresholds, within one more round everyone decides.
    
- Probability of an un-foiled round is a constant > 0.
    

---

### Theorem (Correctness)

- **Validity:** If all start same, tally is huge (≥ 7n/8) → immediate decision.
    
- **Agreement:** Disagreement forces reset; correct king (here: coin) eventually aligns everyone.
    
- **Termination:** Since each round has constant chance of being un-foiled, expected number of rounds until decision is O(1).
    

---

## 🔹 Intuition with Example

Suppose n=9n=9n=9, f=2f=2f=2.

- Correct = 7, Faulty = 2.
    
- Round 1: votes split, faulty try to confuse → no tally ≥ threshold.
    
- Everyone resets to 0.
    
- Round 2: coin = Heads, all adopt 1.
    
- Round 3: tally ≥ 7n/8 = 7.875 → i.e., at least 8 support → decision.
    
- Done in 3 rounds.



# L6:
## 🔹 1. Reliable Broadcast (RB) — Definition

RB is a communication primitive in a system with Byzantine faults. It involves one **sender process** and many **receiver processes**.

**Properties (must hold even if sender is faulty):**

1. **Validity:** If the sender is honest and broadcasts a message mmm, then all honest processes eventually deliver mmm.
    
2. **Agreement:** If one honest process delivers a message mmm, then all honest processes deliver mmm.
    
3. **Integrity:** A message is delivered at most once, and only if it was broadcast by the sender.
    

👉 In short: **all-or-nothing, same message for all honest processes**.

---

## 🔹 2. Equivalence between RB and Byzantine Agreement (BA)

This is a key theory fact.

- **From BA → RB:**
    
    - To implement RB, just let sender propose value mmm.
        
    - Run BA with mmm as input of the sender.
        
    - BA guarantees agreement and validity → you get RB.
        
- **From RB → BA:**
    
    - Each process reliably broadcasts its input value using RB.
        
    - Collect all delivered values.
        
    - Run deterministic choice (e.g., pick the minimum, or majority) to decide.
        
    - RB ensures all honest nodes see the same set of delivered values → so BA achieved.
        

👉 Thus, **RB and BA are polynomially equivalent** (each can simulate the other).  
This is why blockchain protocols often bounce between these abstractions.

---

## 🔹 3. Dolev–Strong’s Reliable Broadcast Protocol

This is the classic **authenticated RB protocol** (uses digital signatures).

**Assumptions:**

- Each message can be digitally signed.
    
- Signatures are unforgeable and verifiable.
    
- Up to fff Byzantine processes, total n>fn > fn>f.
    

---

### Protocol (for sender sss, message mmm)

We run in **rounds 0 to f** (at most f+1f+1f+1 rounds).

1. **Round 0 (sender):**
    
    - Sender signs mmm with its private key → {m}sigs\{m\}_{sig_s}{m}sigs​​.
        
    - Broadcasts to all.
        
2. **Round rrr (for 1≤r≤f1 \leq r \leq f1≤r≤f):**
    
    - If a process receives a message with a chain of rrr valid signatures  
        ({m}sigi0,{}sigi1,…,{}sigir−1)(\{m\}_{sig_{i_0}}, \{ \}_{sig_{i_1}}, …, \{ \}_{sig_{i_{r-1}}})({m}sigi0​​​,{}sigi1​​​,…,{}sigir−1​​​),  
        and it has not seen it before:
        
        - It appends its own signature and forwards to everyone.
            
3. **Decision (after f+1 rounds):**
    
    - Each process checks the messages it received.
        
    - If there is at least one valid chain of signatures on mmm (length f+1f+1f+1), then deliver mmm.
        
    - Otherwise, deliver default value (say ⊥).
        

---

### Why it works

- Byzantine sender might equivocate (send different values to different processes).
    
- But for a value to be delivered, it must collect **f+1f+1f+1 distinct signatures**.
    
- With only fff faulty processes, at least one honest process must have signed.
    
- Once an honest process signs, it propagates → eventually all honest processes see the same chain.
    
- Hence **agreement** and **validity** hold.
    

---

### Complexity

- Runs for f+1f+1f+1 rounds (since faulty nodes could delay their signatures until the end).
    
- Message complexity: O(n2f)O(n^2 f)O(n2f) (because in each round processes forward signed messages).


# L7: Vote + Coin

## 🔹 1. Context

- Deterministic BA requires f+1f+1f+1 rounds or worse.
    
- Randomized BA with a **global coin** achieves **expected constant rounds (Las Vegas)**.
    
- The **Vote + Coin protocol** is the standard example of such an algorithm.
    

We assume:

- nnn processes, up to f<n/3f < n/3f<n/3 Byzantine.
    
- Synchronous model.
    
- Global random coin (all honest see same coin value each round, adversary can’t predict).
    

---

## 🔹 2. Protocol Intuition

Each round has **two parts**:

1. **Vote step:** processes try to see if there’s a clear majority.
    
2. **Coin step:** if no clear majority, everyone flips to the global coin.
    

👉 Over time, the coin ensures convergence; once a majority forms, it locks in forever.

---

## 🔹 3. Protocol Steps

At each process ppp, maintain `vote_p` (initial = input bit).

Repeat rounds until decision:

### (a) **Vote Step**

- Every process broadcasts its current `vote_p`.
    
- Collect all votes into multiset VVV.
    
- Let `maj` = majority value in VVV.
    
- Let `tally` = number of votes for `maj`.
    

Rules:

- If `tally ≥ 2f+1` → adopt `maj`. (strong majority → reliable).
    
- Else → go to coin step.
    

### (b) **Coin Step**

- Global random coin flip C∈{0,1}C \in \{0,1\}C∈{0,1}.
    
- All processes set `vote_p = C`.
    

---

### (c) **Decision Rule**

- If in any round `tally ≥ 2f+1`, then decide on `maj`.
    
- Otherwise continue to next round.
    

---

## 🔹 4. Why it Works

1. **Safety (Agreement):**
    
    - If a value reaches 2f+12f+12f+1 support, then at least f+1f+1f+1 honest nodes voted for it.
        
    - Those honest nodes keep broadcasting it forever.
        
    - So no other value can later reach 2f+12f+12f+1.
        
    - Thus, once decided, never diverges.
        
2. **Termination (Expected constant rounds):**
    
    - In any round with no clear majority, the coin toss aligns all honest nodes to the same value with prob 1/2.
        
    - Geometric distribution → expected 2 rounds until alignment.
        
    - Hence **expected O(1) rounds** to terminate.
        
3. **Validity:**
    
    - If all start with the same value, it already has supermajority in the first round → immediate decision.
        

---

## 🔹 5. Small Example (n=4, f=1)

- Initial votes: P1=0, P2=1, P3=1, P4 (Byzantine).
    
- Round 1 vote step:
    
    - Suppose P4 sends 0 to some, 1 to others.
        
    - P1 sees {0,1,1,0} → 0:2, 1:2 → no 3 votes → coin step.
        
    - P2 sees {0,1,1,1} → majority 1 with 3 votes → decide 1.
        
    - So at least one honest decides.
        
- With probability 1/2, others also align via coin to 1 → full agreement.
    
- If not, continue another round, but expected within 2–3 rounds all converge.
    

---

## 🔹 6. Exam-style Summary

**Vote + Coin protocol:**

- Randomized BA protocol with expected constant rounds.
    
- Each round:
    
    - **Vote step:** adopt majority if ≥ 2f+1.
        
    - **Coin step:** else adopt global coin.
        
- Guarantees:
    
    - **Validity:** unanimous inputs → immediate decision.
        
    - **Agreement:** once 2f+1 achieved, stable forever.
        
    - **Termination:** expected constant rounds (coin ensures convergence).



# L10: Leader Election

## 🔹 1. Motivation

- Many consensus algorithms assume there is a **leader** (or proposer) in each round to drive progress.
    
- In a Byzantine setting, some leaders may be faulty.
    
- So we need a **leader election protocol** that ensures:
    
    - All honest nodes eventually agree on the same leader.
        
    - With high probability, the chosen leader is honest often enough to guarantee progress.
        

---

## 🔹 2. Byzantine Leader Election (BLE) — Definition

**Goal:** elect a leader among nnn processes, tolerating up to f<n/3f<n/3f<n/3 Byzantine.

**Properties:**

1. **Agreement:** All honest processes elect the same leader in a given round.
    
2. **Termination:** Eventually, all honest processes elect some leader.
    
3. **Fairness / Good Leader Probability:** The elected leader is honest with probability ≥ constant (e.g., ≥ 1/2).
    

---

## 🔹 3. Approaches

### (a) Deterministic Leader Rotation

- Trivial method: leader of round rrr = process (r  n)(r \bmod n)(rmodn).
    
- Works, but up to fff rounds may be wasted if those leaders are Byzantine.
    
- Guarantees that eventually a correct leader is chosen.
    

### (b) Randomized Leader Election

- Use a **common coin** to pick a leader uniformly at random from {1,…,n}\{1,\dots,n\}{1,…,n}.
    
- Since fraction of faulty ≤ 1/3, probability of honest leader = ≥ 2/3 each round.
    
- With high probability, progress happens in constant expected rounds.
    
- Used in many modern protocols (e.g., randomized consensus, blockchain protocols).
    

---

## 🔹 4. Example in Blockchain Protocols

- In PBFT-like protocols: leader (called primary) proposes a block. If faulty, next view-change elects new leader.
    
- In randomized protocols: leader chosen via verifiable random function (VRF) or common coin.
    

---

## 🔹 5. Exam-style Summary

**Byzantine Leader Election:**

- Problem: elect a leader process among nnn processes with up to f<n/3f<n/3f<n/3 Byzantine.
    
- Requirements: **agreement, termination, fairness** (leader honest w.p. ≥ constant).
    
- Deterministic approach: round-robin leader rotation (guarantees eventual honest leader after ≤f rounds).
    
- Randomized approach: global coin picks random leader, honest with constant probability.
    
- This primitive is used inside Byzantine Agreement and blockchain protocols to ensure progress despite Byzantine faults.


# L11: Blockchain protocol
## 6.2 Defining a Blockchain Protocol

### What is a blockchain protocol?

- It’s a protocol where a set of nnn nodes agree on a **growing, linearly ordered log of transactions**.
    
- Instead of agreeing one tx at a time, we group them into **blocks** → sequence of blocks = **blockchain**.
    

### Two key properties

1. **Consistency** (safety):
    
    - All honest nodes’ logs are _prefixes_ of each other.
        
    - Means: logs never disagree or shrink, only extend.
        
2. **Liveness**:
    
    - If an honest node receives a transaction txtxtx at round rrr, then by round r+Tconfr+T_{conf}r+Tconf​, every honest node’s log contains txtxtx.
        
    - TconfT_{conf}Tconf​ is the **confirmation time**.
        

👉 Intuition: logs of honest nodes may be at slightly different lengths (because of network delays), but one is always a prefix of the other, and eventually all include the same tx’s.

---

## 🔹 6.3 Blockchain from Byzantine Broadcast (BB)

### Idea

We can **build blockchain** by repeatedly running a **Byzantine Broadcast protocol** (BB).

- Recall: **BB** (also called Byzantine Agreement with a designated sender) ensures that all honest nodes agree on a value broadcast by some sender, even if the sender is faulty.
    
- Here, instead of agreeing on _a single value_, we run BB repeatedly to agree on _blocks_.
    

### Construction

- Time is divided into “epochs” of RRR rounds each.
    
- In epoch kkk (round kRkRkR):
    
    - Designated sender = Lk=(k  n)L_k = (k \bmod n)Lk​=(kmodn).
        
    - That sender proposes a block (transactions not yet in the log).
        
    - Run a new BB instance BBkBB_kBBk​.
        
    - Output = block mkm_kmk​.
        
- Blockchain log at round r is the **concatenation**:
    
    m0 ∥ m1 ∥ ⋯ ∥ mkm_0 \,\|\, m_1 \,\|\, \cdots \,\|\, m_km0​∥m1​∥⋯∥mk​

### Theorem (consistency + liveness)

If the BB protocol tolerates up to fff Byzantine nodes, then the blockchain built by sequential composition satisfies:

- **Consistency**: follows from consistency of each BB.
    
- **Liveness**:
    
    - If tx appears at an honest node in round rrr, then within at most (n+1)(n+1)(n+1) BB instances, that node becomes sender and ensures inclusion of tx.
        
    - So confirmation time =O(Rn)= O(Rn)=O(Rn), where RRR is round complexity of BB.
        

---

## 🔹 6.4 Discussion

- This sequential composition shows feasibility (yes, blockchain can be built from BB).
    
- But not efficient in practice:
    
    - Each block requires a full BB instance → high cost.
        
    - No pipelining across BB runs.
        
- Practical protocols (PBFT, Paxos, Bitcoin) use **direct blockchain constructions**.
    

---

## 🔹 Exam-ready 5-mark summary

**Q:** Explain how a blockchain protocol can be constructed from Byzantine Broadcast.

**A (points):**

- Blockchain = linearly ordered log of blocks; requires **consistency (prefix)** and **liveness (tx included within TconfT_{conf}Tconf​)**.
    
- Construction: sequentially run Byzantine Broadcast (BB) every RRR rounds.
    
    - In round kRkRkR, designated sender Lk=(k  n)L_k = (k \bmod n)Lk​=(kmodn) runs BBkBB_kBBk​.
        
    - Output block appended to the log.
        
- **Consistency:** holds because each BB instance ensures all honest nodes agree on block.
    
- **Liveness:** within O(Rn)O(Rn)O(Rn) rounds, each honest node gets to be sender and includes its pending transactions.
    
- Hence blockchain protocol from BB satisfies both properties, but efficiency is poor.

## 🔹 State Machine Replication (SMR) & Blockchain

- **SMR** = classic problem in distributed systems:  
    _replicate a service across multiple nodes so that all honest nodes see the same sequence of commands (state transitions)._
    
- To do SMR, we need agreement on the order of operations.
    
- A **blockchain protocol is one way to realize SMR**:
    
    - Logs = states.
        
    - Blocks of transactions = commands.
        
    - Consistency & liveness ensure that all honest replicas apply the same sequence.
        

---

## 🔹 Reliable Broadcast vs Byzantine Broadcast

1. **Reliable Broadcast (RB):**
    
    - A primitive where if a sender broadcasts a message, all honest nodes **deliver the same message** (or nothing).
        
    - Properties:
        
        - **Validity:** if sender is honest, everyone delivers the sender’s message.
            
        - **Agreement:** if one honest node delivers a message, all honest nodes deliver the same.
            
        - **Integrity:** no fake messages are delivered.
            
2. **Byzantine Broadcast (BB):**
    
    - RB + ensures agreement even if the sender is faulty.
        
    - Sometimes called _consensus with a designated sender_.
        
    - **BB = stronger primitive** than RB.
        

👉 **Link to blockchain:** If we can build BB, we can repeatedly run it to agree on successive blocks → gives us a blockchain protocol.

---

##
🔹 Blockchain from Byzantine Broadcast

**Construction:**

- Divide time into epochs of RRR rounds.
- In epoch k, node $L_k$ ​= ( k mod n) is the designated sender.
- Run BB instance $BB_k$​: all honest nodes agree on block $m_k$​.
- Append $m_k$​ to the log.
- Output $log = m_0 ∥ m_1 ∥ \dots ∥ m_k$ 
    

**Guarantees:**

- **Consistency:** from BB consistency, all honest logs are prefixes.
- **Liveness:** within at most n+1 epochs, any honest node gets to be sender, so its transactions are included.
- Confirmation time = $O(R_n)$.


