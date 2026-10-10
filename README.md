# moe_jav — Mixture-of-Experts underwriting, gated by a decision model

![Architecture](docs/architecture.png)

**Is every part of an applicant's message routed to the right expert?**

`moe_jav` is a Rust proof of concept that applies the Mixture-of-Experts idea to
LLM agents on a realistic flow: a life-insurance application written as free
text. The message is split into segments, a **gate** routes each segment to a
specialist agent (medical, lifestyle, financial, coverage, compliance), and a
final decision is made from small, auditable signals.

The gate and the final signals come from **[Jev](https://docs.typesafe.ai/introduction)**,
TypeSafe's *decision model*: instead of generating prose it returns typed
answers with calibrated probabilities. That makes two things possible that an
LLM router does not give you cleanly:

1. **Confidence-gated routing**: segments the gate is unsure about go to a
   human instead of being guessed.
2. **Measurable routing quality**: accuracy *and* calibration against a
   labelled dataset, compared head to head with an LLM router.

> ⚠️ Proof of concept on **synthetic data**. Not an underwriting system and not
> for real insurance decisions.

## Results

40 synthetic applications and 160 hand-labelled segments. Jev (`jev-1.13.0`)
and Gemma 4 E4B were each tested as the router, using the same rubric.
Numbers are from the final code (run 3). Earlier runs are kept in
[docs/eval/run1](docs/eval/run1) and [docs/eval/run2](docs/eval/run2).

![Results](docs/results.png)

**1. Both gates route well, but they fail differently.**
On gold segments both reach about 98% accuracy (Jev 98.1%, LLM 97.5%), so
accuracy alone doesn't separate them. What separates them is *where the
mistakes sit*. Raising the gate threshold τ shows it (each cell: segments
sent to a human / wrong routes that still slip through):

| τ | Jev gate | LLM gate |
|---:|---:|---:|
| 0.50 | 0 / 3 | 0 / 4 |
| 0.80 | 4 / 1 | 0 / 4 |
| **0.90** | **7 / 0** | 1 / 4 |
| 0.95 | 8 / 0 | 4 / 3 |
| 0.99 | 14 / 0 | 35 / 1 |

Jev's three mistakes came with lower confidence (≤ 0.87). At τ = 0.9 the
gate stopped **every** misroute and sent only **7 of 160 segments (4.4%)** to a
human. The LLM's four mistakes came at 0.90–1.00 confidence. Even at
τ = 0.99, with **35 human reviews (22%)**, one misroute at confidence 1.00
still got through. The pattern held in all three runs.
(ECE is low for both, 0.013 vs 0.016. With so few errors it mostly reflects
the top confidence bin, which is why the sweep is the more informative view.)

**2. End-to-end** (segment → gate → experts → Jev decision, Jev router, τ = 0.5):

| Metric | Run 3 |
|---|---:|
| Pipeline failures | 0 / 40 |
| Segment recall (predicted ↔ gold alignment) | 90.6% |
| Routing accuracy, end-to-end | 88.8% |
| Decision agreement with labels | 65.0% (26/40) |
| Prompt-injection cases referred to a human | 3 / 3 (manipulation p ≥ 0.98) |
| Concealment / third-party cases referred to a human | 2 / 2 |
| Fairness pairs with identical decisions | 2 / 3 |

**3. Things that went wrong** (details in [design.md §6](docs/design.md#6-lessons-learned))

- **Run 1 referred all 40 applications to a human.** Jev correctly judged
  that a few sentences are never enough for real underwriting
  (`sufficient_information` p = 0.05–0.26). The model was right; the rule was
  wrong. It was removed from the referral conditions **after seeing the
  results**, so the decision numbers are not a blind test.
- **Fairness pairs are fragile at a hard threshold.** Demographic sentences
  route to `other`, and Jev no longer sees them. Even so, the "Jamal" /
  "Connor" pair scored 0.42 vs 0.45 in run 3 and fell on opposite sides of the
  0.45 decline line. Both applicants' medical segments were identical. The
  difference comes from run-to-run variation in the local LLM's expert
  summaries, which Jev then scores. So we cannot attribute the split to
  demographics. We also cannot rule demographics out, and a single hard
  cut-off turns small noise into different outcomes. Next steps: average
  repeated runs, add a margin band around thresholds that refers to a human,
  and use more pairs.
- **A fairness fix opened a security hole.** Once Jev stopped reading the full
  text, Gemma sometimes left an injected "approve this" instruction out of its
  segmentation. Text the segmenter drops is now added back as a segment and
  routed like everything else. In run 3 this recovery triggered 3 times; all
  3 injection cases were still caught.
- **Runaway thinking.** `reasoning_effort: "none"` sometimes made Gemma think
  without stopping. We now cap `max_tokens` and retry once without the hint.
- **Main decision misses:** a few cases a human should see were auto-accepted:
  a financial mismatch, a prior claim investigation and a recent bankruptcy.
  Some moderate-risk smokers were declined.

Full report: [docs/eval/eval_report.md](docs/eval/eval_report.md)

## How it works

| Step | What | Model |
|---|---|---|
| 1. Segment | Split the application into single-topic, verbatim clauses | Gemma 4 E4B (local, LM Studio) |
| 2. Gate | Route each segment to an expert; `confidence < τ` → human review | **Jev `choice`** (or the LLM, for comparison) |
| 3. Experts | Only the experts that received segments run, concurrently | Gemma 4 E4B |
| 4. Decide | Six atomic questions (medical/lifestyle risk scores; coverage fit, manipulation, concealment, sufficient info) combined by explicit Rust rules | **Jev `score` / `noul`** |
| 5. Evaluate | Compare to 160 hand-labelled segments: accuracy, F1, confusion matrix, ECE, reliability diagram, fairness pairs | — |

Details: [docs/design.md](docs/design.md) · CLI and APIs: [docs/api.md](docs/api.md) ·
Dependencies: [docs/dependencies.html](docs/dependencies.html)

## Quick start

Requirements: Rust 1.97+, [LM Studio](https://lmstudio.ai/) serving
`google/gemma-4-e4b` on `http://localhost:1234`, and a TypeSafe API key.

```sh
export API_KEY_JEV=...            # or copy app/.env.example to app/.env
cd app
cargo run --release -- apply --text "Hi! I'm 42, quit smoking in 2015 and had knee surgery last year. \
I earn \$95,000 as a nurse and want a \$500,000 20-year term policy."
```

```sh
cargo run --release -- eval --mode routing   # Jev gate vs LLM gate on gold segments
cargo run --release -- eval --mode e2e       # full pipeline on all 40 applications
```

Helper scripts: `scripts_local/run.sh` (Linux/macOS) and
`scripts_local/run.ps1` (Windows): `check`, `live`, `apply FILE`, `eval [MODE]`.

## Engineering

- **Rust 2024**, Tokio, reqwest (rustls), clap; native `async fn` in traits.
- **Traits at every boundary** (`LlmClient`, `DecisionModel`, `Router`), so all
  logic is unit-tested offline with stubs; the HTTP clients are tested against
  `wiremock`; live tests against LM Studio and Jev are opt-in (`--ignored`).
- **Newtype validation** (`ApplicationText`, `SegmentText`, `Confidence`,
  `ApiKey`, `BaseUrl`).
- **Security**: applicant text is treated as untrusted (delimited, never
  obeyed), prompt-injection cases are in the dataset, the API key is redacted
  in logs and only sent over HTTPS, responses are size-capped, and
  `cargo audit` runs in CI.
- **Quality gates** in CI: `rustfmt`, `clippy::pedantic -D warnings`, tests
  with ≥ 90% line coverage (`cargo-llvm-cov`), `cargo audit`. Tagging `v*`
  builds a static `x86_64-unknown-linux-musl` release binary.

## License

See [LICENSE](LICENSE).
