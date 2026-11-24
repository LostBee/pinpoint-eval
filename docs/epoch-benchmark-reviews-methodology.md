# Epoch AI Benchmark Reviews: Methodology, Audits & X.com Insights

> **Research Date:** November 24, 2026  
> **Primary Sources:** [Epoch AI Methodology](https://epoch.ai/data/benchmark-reviews-documentation/methodology), [Overview](https://epoch.ai/data/benchmark-reviews-documentation), [Included Benchmarks](https://epoch.ai/data/benchmark-reviews-documentation/included-benchmarks), [FAQ](https://epoch.ai/data/benchmark-reviews-documentation/faq), [Benchmark Registry](https://epoch.ai/benchmarks/search?reviewed=verified&reviewed=flawed&reviewed=not-enough-info), and `@EpochAIResearch` / Epoch researcher threads on X.com.

This document summarizes how Epoch AI decides whether an external AI benchmark is **Verified**, **Flawed**, or **Not enough info**—to serve as a reference when hardening `pinpoint-eval` against independent benchmark auditing standards.

---

## 1. Decision Framework (`Rubric v1`)

Epoch AI evaluates external benchmarks sequentially across three sections (`Rubric v1`, expected to evolve as new failure modes emerge) and assigns one of four labels on `https://epoch.ai/benchmarks`:

| Verdict | Definition | What Gets Published |
|---|---|---|
| **Verified** | Section 1 (*Reviewability*) is **Full** or **Partial**, and **none** of the four Section 2 criteria cross the defect threshold. | A full review including Section 3 (*Evaluation Quality*) and a *Limitations* section logging minor caveats. |
| **Flawed** | Section 1 is **Full** or **Partial**, and **at least one** Section 2 criterion crosses the defect threshold. | A focused/partial write-up detailing the specific disqualifying defect(s). **Review stops once the threshold is crossed.** |
| **Not enough info (NEI)** | Section 1 is **Inadequate**—required tasks, scoring logic, or harness/API settings are inaccessible. | Badge only on the benchmark page (no `/review` page). |
| **By Epoch** | Benchmark created by Epoch AI. Excluded due to conflict of interest. | No Epoch review; external audits welcomed. *(Epoch notes this does not imply error-free work, citing an audit of FrontierMath v1 that found 42% of problems contained errors.)* |

---

### Section 1: Reviewability (Gatekeeper → `Not enough info` if Inadequate)

Before auditing quality, Epoch checks whether the benchmark is inspectable:

| Level | Meaning | Status |
|---|---|---|
| **Full** | **All** tasks and scoring logic are inspectable (publicly, **or privately to reviewers under NDA**), **and** harness/API settings used for each model (reasoning effort, token/time limits, tool access, system prompts) are fully disclosed. | `proceed to 2` |
| **Partial** | A **representative sample** of tasks and scoring logic is inspectable, with **full** harness/API settings disclosed. | `proceed to 2` |
| **Inadequate** | Limited, biased, or no inspectable tasks or scoring logic, **or** harness/API settings undisclosed. | `stop → NEI` |

**Private Benchmark & Outreach Process:**
- Epoch contacts developers of private benchmarks to request confidential access (under NDA if needed) and gives them **one week** to respond with a timeline for sharing materials before applying the `Not enough info` label (withholding publication if timely access is expected).
- To prevent contamination/leakage on private benchmarks, Epoch discloses *what* was evaluated and includes only generic information (e.g., scoring error types without task-level analysis), and redacts specific exploit details when requested by the creator (as done for `Vending-Bench 2`).

---

### Section 2: Minimum Standards (Fail Any One → `Flawed`)

To be **Verified**, a benchmark must pass **all four** items. Failing any one triggers `Flag [stop → Flawed]`:

#### 1. Scoring
- **Examples of defects:**
  - *Impossible as written:* Underspecified task, hidden requirement, missing file/tool.
  - *False Negative:* Overly strict scorer, stale/incorrect ground truth, **dependent on live external state that can drift**, or sandbox failure independent of the agent.
  - *False Positive:* Lax scorer, reward-hackable environment, or stated skill bypassed via shortcut (e.g., exploiting scoring logic or retrieving the answer from the harness/web).
  - *Severe misalignment:* Task egregiously doesn't measure the claimed capability.
- **Default threshold for Flawed:**
  - Either errors affect **≥ 20%** of the total score in the inspected sample, **OR**
  - There is an issue that corrupts grading at scale.
- **20% Threshold Math (Footnote 2):**
  - *Equally weighted all-or-nothing tasks:* ≥ 20% of tasks have defects.
  - *Partial scoring or unequal task weights:* ≥ 20% of the total possible score is affected by defects.
  - *Single-task benchmarks:* Defects could affect the reported score by ≥ 20%.
- **Sampling Protocol (Footnote 3):**
  - **≤ 50 available tasks:** Assess **all** available tasks.
  - **> 50 available tasks:** Draw a **random sample of 50** tasks (stratified by category, when present).
  - **Borderline band (15%–25% observed error rate):** Expand the sample to **100 tasks**.
- **External Audits Rule:** For benchmarks with existing credible reviews, Epoch cites the external review rather than replicating the findings (e.g., OpenAI's Feb 2026 audit of `SWE-bench Verified`; Ansari et al. 2026 & Anthropic's system card for `CritPt v1`).

#### 2. Benchmark Consistency
- **Examples:** Scorer, instructions, or ground truth changed without a version bump.
- **Default threshold for Flawed:** Leaderboard has incomparable results from different versions.
- **Practical materiality:** Minor edits with negligible impact do not auto-fail (e.g., `SimpleQA Verified` reworded 1/1,000 questions without a version bump—0.1-pt impact—and passed). In contrast, `ARC-AGI-3` changed prompts to allow carry-forward context without a version bump or re-running older entries and was flagged.

#### 3. Elicitation
- **Examples:** Model elicitation is *extremely constraining* and is **not** the focus of the benchmark:
  - Under-resourced relative to task size (token/turn/time limit, sandbox resources).
  - Poor context management.
  - Lack of agentic environment where one would be natural to provide.
  - Excessive non-voluntary termination for agentic benchmarks.
- **Default threshold for Flawed:** Substantially reduced performance compared to reasonable alternatives for the tasks.

#### 4. Bias in Evaluation Setup
- **Examples:** Uneven compute/token budgets; unfair scaffold choice (e.g., only a subset of models optimized).
- **Default threshold for Flawed:** Material model-specific advantage found.

---

### Section 3: Evaluation Quality (Aspirational / Non-Disqualifying)

Logged in full on **Verified** benchmarks (typically left `Not reviewed` on **Flawed** benchmarks that stop at Section 2). Omitting or failing these is not automatically disqualifying, but surfaces caveats:

| Question | Possible Statuses |
|---|---|
| **Elicitation and resource adequacy:** Are resources (reasoning token/turn budget, tool access, etc.) sufficient for models to perform near their ceiling? | `Sufficient` / `Constraining` / `Unreasonably constraining` / `Unknown` / `Not reviewed` |
| **Scaffold fairness:** What scaffold does the leaderboard report? | `Shared common scaffold` / `Mix of model-specific and common scaffolds` / `Model-specific scaffolds` / `Not reviewed` |
| **Contamination risk:** Is there evidence/risk of contamination? | `—% of tasks public`, `—% of solutions public` (as of review date) / `Not reviewed` |
| **Human completability:** Has human completability been assessed? | `All tasks` / `Representative set of tasks` / `Poor implementation` / `Not established` / `Not reviewed` |
| **Score range:** (if possible to estimate) | `— floor, — ceiling` / `Not reviewed` |
| **Statistical adequacy:** How many runs/model (**≥ 5 recommended** for error bars) | `— runs/model` / `Unknown` / `Not reviewed` |
| **Construct validity:** | `Measures stated capabilities` / `Partially measures stated capabilities` / `Does not measure stated capabilities` / `Not reviewed` |

*(In Epoch's accompanying analysis ["Why benchmarking is hard"](https://epoch.ai/gradient-updates/why-benchmarking-is-hard), scaffold differences alone caused up to 11% variance for GPT-5 and 15% for Kimi K2 Thinking.)*

---

## 2. Operational, Versioning & Creator-Response Policies

- **Prioritization:** Benchmarks are selected based on **Impact** (tasks Epoch cares about / safety-relevant capabilities), **Reach** (widely cited, in recent system cards, or underrated), and **Diversity** (under-covered capabilities). Domain-knowledge benchmarks take longer because Epoch hires external subject-matter experts.
- **Static snapshot rule:** Reviews bind to a specific version snapshot. Fixing bugs in a newer release does not retroactively change an older version's verdict; Epoch considers re-reviewing new versions subject to reviewer capacity.
- **Stop-on-threshold (partial audit) rule:** Once defects exceed the Flawed threshold, Epoch stops inspecting—so a Flawed review is not an exhaustive catalog of every error.
- **Pre-publication notice & creator responses:** Epoch does not hold publication for fixes, reaches out to developers **at least one day prior to publication**, and offers to link/publish the developer's full response on the review page (`reviews@epoch.ai`).

---

## 3. Empirical Track Record (30 Reviewed Benchmarks, Aug–Nov 2026)

Out of 100 benchmarks tracked in Epoch's registry as of November 2026, **30** have been reviewed: **6 Verified (20%)**, **15 Flawed (50%)**, and **9 Not enough info (30%)**.

### Verified (6)
1. **SimpleQA Verified** (Sep 10, 2026) — Drew topic-stratified sample of 50/1,000; found 5/50 (10%) defective (below 20% threshold and outside the 15–25% expansion band).
2. **ExploitBench v0.1** (Sep 12, 2026) — Passed all Section 2 criteria; logged contamination and resource constraints in Section 3.
3. **WeirdML v2** (Aug 10, 2026)
4. **PostTrainBench v1.1** (Sep 9, 2026)
5. **GraphWalks** (Sep 29, 2026)
6. **MazeBench v1.1** (Oct 27, 2026)

### Flawed (15)
- **Batch 1 (Aug–Sep 2026):** `Terminal-Bench 4.0.0` (45.5% broken tasks), `SWE-bench Verified` (cited OpenAI audit: ≥16.4% floor / likely ≥20% flawed tests + RDI reward hack + contamination; Construct validity marked *Does not measure stated capabilities*), `SWE-Bench Pro`, `Humanity's Last Exam` (46% of 48 sampled questions broken), `DeepSWE v1.1` (global grading bug + 23/131 false negatives), `TextQuests`, `Lech Mazur Writing`, `BFCL v4`, `HealthBench Professional`.
- **Batch 2 (Sep–Oct 2026):** `OSWorld 2.1`, `CritPt v1` (cited Ansari et al. 29.6% flawed and Anthropic's *CritPT-Corrected* 43.7% revised), `CMT-Benchmark v1.0.0`.
- **Batch 3 (Oct 2026):**
  - `ARC-AGI-3` — Failed all 4 Section 2 items: models not instructed to optimize action efficiency, upper-median human baseline + 1.15 cap, May 20, 2026 prompt change without version bump, Standard harness discarding reasoning and trimming history (GPT-6 Astra 62.7% vs. 98.6% on Provider Adapter), and \$10k/run cap exceeded up to \$49,800.
  - `Vending-Bench 2` — Passed Scoring, Consistency, and Elicitation, but failed **Bias in Evaluation Setup**: `tiktoken` counted encrypted reasoning strings as raw text, shrinking OpenAI's effective context window (~45k vs. 69k) more severely than Anthropic's, plus Bedrock vs. OpenAI tool-call enforcement differences and live web-search drift. (Also vulnerable to >\$1M exploit when prompted to exploit, vs. \$16,484 leaderboard top.)
  - `Agents' Last Exam v1.1` — ≥ 44% of sampled tasks had severe errors.

### Not Enough Info (9)
`FrontierCode 1.1`, `FrontierCode 2.0`, `CritPt v2`, `CursorBench 4.0`, `DeepRSI`, `Chartography`, `TasteVal`, `Remote Labor Index`, `Surface Evolver Bench`.

---

## 4. Additional Insights from X.com Threads

1. **Release waves & capacity bottleneck:**
   - **Sept 17, 2026** ([launch thread](https://x.com/EpochAIResearch/status/2100704765332394255)): 15 benchmarks (4 Verified, 9 Flawed, 2 NEI).
   - **Oct 23, 2026** ([batch 2](https://x.com/EpochAIResearch/status/2113703991767105607)): +4 reviews (1 Verified, 3 Flawed) + 4 NEI; highlighted reviewer capacity as the main bottleneck.
   - **Nov 23, 2026** ([batch 3](https://x.com/EpochAIResearch/status/2124945252725510614)): +4 reviews (1 Verified, 3 Flawed) + 4 NEI.
2. **Scoring defects & grader exploits are the dominant failure mode:**
   - Epoch researcher Michelle Campeau (quoted via `@MTSlive`: [1](https://x.com/MTSlive/status/2101088613899641082), [2](https://x.com/MTSlive/status/2101080402538758369)) noted that false positives/negatives in answer keys and sandbox/grader escapes (e.g., agents writing a success byte directly instead of solving the task) account for most Flawed verdicts—and warned that broken tasks induce reward hacking during RL training.
3. **Private lab "corrected" forks distort public comparisons:**
   - Campeau ([3](https://x.com/MTSlive/status/2101092525163643123)) pointed to `CritPt`, where Anthropic privately fixed >40% of broken questions (`CritPT-Corrected`), making score jumps appear to be +40% capability gains rather than benchmark bug fixes.
4. **Epoch audits the version actually cited on leaderboards:**
   - When `HLE` creators (`@longphann_`, `@notRichardRen`) noted bugs were fixed in `HLE-Rolling`, Epoch's Alexander Barry (`@AlexBarry4`, [post](https://x.com/AlexBarry4/status/2100735519147647464)) and Yafah Edelman (`@YafahEdelman`, [post](https://x.com/YafahEdelman/status/2100734129046458684)) clarified that Epoch reviews the version actively cited on public leaderboards and system cards.
5. **Primary leaderboard metric & literal wording matter (`TextQuests` debate):**
   - In response to `TextQuests` creators (`@MantasMazeika96`, `@longphann_`, `@notRichardRen`), `@AlexBarry4` ([post](https://x.com/AlexBarry4/status/2100962067960238587)) explained Epoch audits the primary/first-column leaderboard metric (`Progress (%)`) against its literal stated description, acknowledged Rubric v1 was designed primarily for per-question tasks rather than nonlinear game checkpoints, and committed to issuing a public correction if a Flawed call is later found to be wrong.
6. **Binary `Flawed` granularity tension:**
   - Greg Burnham (`@GregHBurnham`, [post](https://x.com/GregHBurnham/status/2100733908677496942)) emphasized *"no shade on creators"* given how hard ambitious benchmarks are. Community critics (e.g., `@NickEMoran`, [reply](https://x.com/NickEMoran/status/2124945288746225771)) noted a key tension in Rubric v1: `Flawed` lumps together *"45%+ of ground-truth tasks are broken"* and *"the scoring/instruction design conflates capability with propensity"* under the same label.

---

## 5. Initial Implications for `pinpoint-eval`

`pinpoint-eval` already aligns with several Epoch criteria (frozen sets with pre-run SHA-256 commitments, rotation guard against future model contamination, unified Inspect AI scaffold, explicit `reasoning.effort` tracking). Key areas to solidify against Epoch's rubric later:

1. **Reviewability (Avoid `Not enough info`):**
   - Since ground-truth images/coordinates are private, provide either a **public representative sample split** (enabling *Partial* reviewability) and/or a documented **confidential auditor/NDA export package**, alongside full public disclosure of scoring logic, prompts, OpenRouter parameters, token/time limits, and Tavily Track B settings.
2. **Scoring, Consistency, Elicitation & Bias (Avoid `Flawed`):**
   - *Scoring:* Audit ground-truth coordinate precision (camera position vs. depicted landmark), border/continent edge cases, parser/refusal false negatives, and **Track B live web-search drift** (external state drift is an explicit Epoch trigger).
   - *Consistency:* Enforce strict version bumps on any change to prompts, scorer, image sanitization, or Tavily parameters—never mix results on one leaderboard.
   - *Elicitation & Bias:* Guard against OpenRouter provider routing differences, silent API truncations/timeouts counted as wrong answers, uneven token/reasoning budgets, and ensure models are explicitly instructed on the exact scoring objective.
3. **Evaluation Quality (Section 3):**
   - Support **≥ 5 runs/model** (especially for stochastic Track B runs) to report error bars, establish a **human completability baseline** on a representative subset, and document score floor/ceiling and strict vs. refusal-adjusted metrics.
