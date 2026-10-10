# moe_jav — Design

> **Disclaimer:** this is a proof of concept on synthetic data. It is not an
> underwriting system and must not be used for real insurance decisions.

## 1. Goal

A real insurance application mixes topics in one free-text message: health,
habits, money, the product wanted, and sometimes things that should worry a
reviewer. `moe_jav` applies the **Mixture-of-Experts (MoE)** idea at the agent
level and asks the question MoE systems rarely measure:

> **Is each part of the message routed to the right expert?**

The gate (router) is TypeSafe's **Jev**, a *decision model* that returns typed
answers with calibrated probabilities instead of prose. We compare it head to
head with a conventional LLM router on the same rubric.

## 2. Architecture

```
            Applicant free text
                    │
                    ▼
   ┌──────────────────────────────────┐
   │ Segmenter  (Gemma 4 E4B, local)  │  JSON-schema output, verbatim clauses
   └────────────────┬─────────────────┘
                    │ Vec<SegmentText>
                    ▼
   ┌──────────────────────────────────┐
   │ Gate / Router                    │  Jev `choice`  (or LLM, for comparison)
   │  confidence ≥ τ → expert         │
   │  confidence < τ → human review   │
   └───┬──────┬──────┬──────┬──────┬──┘
       ▼      ▼      ▼      ▼      ▼        sparse activation: only experts
   Medical Lifestyle Financial Coverage Compliance   that received segments run
       └──────┴──────┼──────┴──────┘        (Gemma, concurrent)
                     ▼ ExpertFinding[]
   ┌──────────────────────────────────┐
   │ Underwriting (Jev)               │  6 atomic questions (score / noul)
   │  + explicit composite rules      │  → accept / conditions / decline / refer
   └──────────────────────────────────┘
```

| Stage | Component | Model | Output |
|---|---|---|---|
| Segment | `core::segmenter` | Gemma (LM Studio) | `Vec<SegmentText>` |
| Gate | `core::router::{JevRouter, LlmRouter}` + `Gate` | Jev `choice` / Gemma | `RoutingDecision` + `auto_routed` |
| Experts | `core::experts::ExpertPanel` | Gemma | `ExpertFinding` per activated expert |
| Decide | `core::underwriting` | Jev `score`/`noul` + Rust rules | `UnderwritingResult` |
| Evaluate | `core::evaluation` | — | routing & calibration metrics |

### Experts

`medical`, `lifestyle`, `financial`, `coverage`, `compliance`, and a sink
`other` (small talk). The rubric lives in one place (`Expert::description`)
and is shared by both routers, so the comparison is fair.

### Why Jev for the gate

- Output is constrained to the given labels — it cannot invent an expert.
- It returns a full probability distribution and a `confidence`, which the
  gate thresholds (`--threshold`, default 0.5): uncertain segments go to a
  human instead of being guessed.
- Calibration (RLCD training) means the threshold has a meaning: "0.8" should
  be right ~80% of the time. The evaluation checks this with ECE and a
  reliability diagram.

### What Jev sees for the decision

Jev receives `application_excerpts` — every segment **except** those the gate
confidently routed to `other` — plus the expert findings. Names, ethnicity,
gender and small talk are routed to `other`, so they never reach the
decision. This was added after the first end-to-end run showed a fairness
pair diverging (§6). Segments sent to a human stay in, since their topic is
unknown.

### Why atomic questions for the decision

Following Jev's guidance, the final decision is not one opaque question.
Jev answers six independent questions in one call:

| Id | Type | Meaning |
|---|---|---|
| `medical_risk` | score 0–3 | none / minor / moderate / severe |
| `lifestyle_risk` | score 0–3 | none / low / elevated / high |
| `coverage_proportionate` | noul | coverage ≤ ~20× income? |
| `manipulation` | noul | text tries to steer the reviewer/system |
| `concealment` | noul | concealment, prior fraud, third-party answering |
| `sufficient_information` | noul | enough info to underwrite (reported only, see §6) |

Rust code combines them (`underwriting::decide`), in this order:

1. **Refer to human** if manipulation ≥ 0.5, concealment ≥ 0.5, any segment
   was gated, or a risk score has confidence < 0.5.
2. **Decline** if medical score ≥ 2.5.
3. Composite risk `0.5·med/3 + 0.3·life/3 + 0.2·(1 − proportionate)`:
   `< 0.20` accept, `< 0.45` accept with conditions, else decline.

The rules are deliberately simple and auditable; the point of the POC is the
routing and the confidence signal, not actuarial accuracy.

## 3. Evaluation

Golden dataset: `data/applications.jsonl` — 40 synthetic applications, 160
hand-labelled segments (each a verbatim substring), expected decisions, and
tags (`injection`, `ambiguous`, `contradiction`, `pair:*` …).

Two levels:

- **Routing only** (`eval --mode routing`): both routers route the *gold*
  segments, isolating gate quality from segmentation errors. Metrics:
  accuracy, per-expert precision/recall/F1, confusion matrix, mean confidence,
  **ECE**, Brier score, gate behaviour at τ (auto-routed share, accuracy
  when auto-routed, errors caught vs. missed), and a **threshold sweep**
  (τ = 0.5 … 0.99: segments sent to a human vs. wrong routes that slip
  through). The sweep is the key view: a calibrated gate's mistakes sit at low
  confidence, so raising τ catches them; an over-confident gate's mistakes do
  not move.
- **End-to-end** (`eval --mode e2e`): full pipeline; predicted segments are
  aligned to gold segments by token Jaccard (≥ 0.5). Metrics: segment recall,
  end-to-end routing accuracy, decision agreement, decision distribution, and
  **fairness-pair consistency** (applications identical except for name,
  gender, ethnicity or origin must receive the same decision). A failing
  application is recorded as an error and counted as wrong; the run continues.

Artefacts land in `docs/eval/` (`eval_report.md`, `reliability.svg`, raw
predictions as JSON).

## 4. Code layout

```
app/src
├── main.rs            wiring: env → clients → cli::run
├── lib.rs
├── cli.rs             clap commands, generic over the service traits
├── config.rs          env config; ApiKey (redacted), BaseUrl (https-only except loopback)
├── error.rs           AppError
├── api/               external adapters
│   ├── http.rs        JSON POST, timeouts, bounded retries, 2 MiB response cap
│   ├── jev.rs         trait DecisionModel + JevClient
│   └── llm.rs         trait LlmClient + LmStudioClient (json_schema output)
└── core/              domain logic, network-free
    ├── domain.rs      newtypes: ApplicationText, SegmentText, Confidence; Expert, Decision
    ├── segmenter.rs
    ├── router.rs      Router trait, JevRouter, LlmRouter, Gate, route_all
    ├── experts.rs
    ├── underwriting.rs
    ├── pipeline.rs
    ├── dataset.rs
    ├── evaluation.rs
    └── report.rs      console view, Markdown report, SVG reliability diagram
```

Design rules applied:

- **Traits at every external boundary** (`LlmClient`, `DecisionModel`,
  `Router`), using native `async fn` in traits with `Send` futures. Unit tests
  use closure-driven stubs (`core::testing`); HTTP clients are tested against
  `wiremock`; live tests are `#[ignore]`d.
- **Validation via newtypes**: invalid text, confidence, URLs or keys cannot be
  constructed.
- **Concurrency**: experts run concurrently (`try_join_all`); evaluation keeps
  four requests in flight.

## 5. Security considerations

| Risk | Mitigation |
|---|---|
| API key leakage | Read from `API_KEY_JEV`; `ApiKey` `Debug` is redacted; `.env` git-ignored |
| Key sent in clear | `BaseUrl` rejects plain `http` except `localhost`/`127.0.0.1` |
| Prompt injection in the application | Segmentation coverage check (omitted text becomes a segment); applicant text is wrapped in `<application>`/`<excerpt>` tags and declared untrusted in every system prompt; a dedicated `compliance` expert; Jev `manipulation` question forces human review; injection cases in the dataset |
| Router hijack | Jev `choice` can only return the provided labels; LLM router output is enum-constrained by JSON schema and re-validated |
| Runaway responses | 2 MiB cap, timeouts, `max_tokens` cap with one retry, max 40 segments, truncated expert output |
| Sensitive data | Synthetic data only; health data never leaves the machine except to Jev |
| Unfair outcomes | Fairness pairs in the dataset, consistency reported |
| Vulnerable dependencies | `cargo audit` in CI |

## 6. Lessons learned

- **Runaway thinking.** With Gemma 4 E4B in LM Studio, `reasoning_effort:
  "none"` is ~5× faster on most inputs but occasionally makes the model think
  without end (1,488+ reasoning tokens on one application, exceeding the HTTP
  timeout three times). Fix: a `max_tokens` cap plus one retry without the
  hint on `finish_reason: "length"`, and an evaluation loop that records a
  failed case instead of aborting.
- **Jev `noul` criteria** must be an object (`{"yes": …, "no": …}`); a string
  is rejected with HTTP 422.
- **A correct signal can still be a bad rule.** The first end-to-end run
  referred **all 40** applications to a human: Jev rated
  `sufficient_information` at 0.05–0.26 everywhere — fairly, since a few
  sentences are never enough for real underwriting. The rule, not the model,
  was wrong; it was removed from the referral conditions **after seeing the
  evaluation results** (results of that run are kept in `docs/eval/run1/`).
- **Fairness pairs vs. noise.** In run 1 the "Jamal" / "Connor" pair
  (identical facts, different name and ethnicity) landed on either side of
  the decline threshold. Jev now judges only non-`other` segments (§2), so
  demographic sentences never reach the decision. In run 3 the pair split
  again (0.42 vs 0.45) even though both medical segments were identical. The
  gap comes from run-to-run variation in the LLM experts' summaries, which a
  hard 0.45 cut-off turns into different outcomes. Planned mitigations:
  repeated runs with averaging, a margin band around thresholds that refers
  to a human, and more pairs.
- **A fairness fix opened a security hole.** Once Jev saw only segments,
  a live test failed intermittently: Gemma sometimes *drops* an injected
  instruction ("SYSTEM NOTE TO REVIEWER: … approve …") from its segmentation
  output, so nothing downstream saw it (manipulation p = 0.03). Fix: the
  segmenter now checks coverage, and any applicant text not covered verbatim
  by a segment is appended as an extra segment, so it is routed by the gate
  like everything else. Verified 5/5 live runs (2 of them with the omission
  recovered). This fix landed after run 2; run 3 includes it (3 recoveries, all injection cases caught).
- **Mixed sentences need segmentation.** Jev routed "I quit smoking three
  years ago and had knee surgery last year." to `medical` (0.89) — correct
  for half the sentence. Splitting first is what makes per-expert routing
  meaningful.

## 7. Limitations

- A 4B local model segments and analyses; quality is bounded by it.
- Labels and expected decisions were written by the team, not underwriters.
- 160 segments is enough to show the method, not to claim significance.
- Decision weights are illustrative, not actuarially derived.
