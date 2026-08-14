# Results

All numbers on **LibriSpeech-PC test-clean cross-sentence** (1127 utterances, the F5-TTS official protocol), F5-TTS v1 base, 32 ODE steps, CFG 2.0, sway sampling −1.0. WER/CER in %, corpus-level (total edits / total reference words) after F5-TTS official text normalization.

Evaluator aliases and their ASR lineages:

| Alias | Checkpoint | Family |
|---|---|---|
| `fwhisper-lgv3` | Whisper-large-v3 via faster-whisper — **official F5-TTS evaluator** | Whisper |
| `w2v2-lv60` | `facebook/wav2vec2-large-960h-lv60-self` | wav2vec 2.0 |
| `hubert-lg` | `facebook/hubert-large-ls960-ft` | HuBERT |
| `distil-sm` | `distil-whisper/distil-small.en` | Whisper |
| `whisper-med` | `openai/whisper-medium.en` | Whisper |

Verifier aliases: `w2v2-base` = `facebook/wav2vec2-base-960h` (95M), `distil-sm` = `distil-whisper/distil-small.en` (166M), `distil-v3` = `distil-whisper/distil-large-v3` (756M). `ens3` = rank-average of all three.

## 1. Baseline reproduction

Single-shot F5-TTS reaches **2.06% WER** under the official evaluator, against the **2.42%** the F5-TTS paper reports for the same 32-NFE configuration on this subset. We are slightly *better* than published rather than worse, which is the direction a faithful reproduction is allowed to land in — the residual comes from decoding and text-normalization details, not from a different system. The pipeline is reproducing the published model, not a degraded variant of it.

## 2. Headline: N=3

| Method | WER | CER | RTF | rel. | p |
|---|---|---|---|---|---|
| F5-TTS (baseline) | 2.06 | 0.71 | 0.190 | — | — |
| BoN `w2v2-base` | 1.99 | 0.62 | 0.383 | −3.5% | 0.31 |
| BoN `distil-sm` | 2.04 | 0.59 | 0.388 | −1.0% | 0.93 |
| **BoN `distil-v3`** | **1.88** | **0.53** | 0.379 | **−8.7%** | **0.030** |
| BoN `ens3` | 1.91 | 0.54 | 0.429 | −7.2% | 0.057 |

`p` is a two-sided paired permutation test (10,000 permutations) against the baseline under `fwhisper-lgv3`.

## 3. The confound: scaling N under three evaluators

| Method (N) | fwhisper-lgv3 | w2v2-lv60 | hubert-lg | mean | p |
|---|---|---|---|---|---|
| F5-TTS (baseline) | 2.06 | 1.52 | 1.92 | 1.83 | — |
| `w2v2-base` (3) | 1.99 | 1.41 | 1.74 | 1.71 | 0.05 |
| `w2v2-base` (5) | 2.20 ↑ | 1.41 | 1.75 | 1.79 | 0.97 |
| `w2v2-base` (10) | 1.98 | **1.40** | 1.72 | 1.70 | 0.08 |
| `distil-v3` (3) | 1.88 | 1.45 | 1.74 | 1.69 | <.001 |
| `distil-v3` (5) | 1.80 | 1.48 | 1.75 | 1.68 | <.001 |
| `distil-v3` (10) | **1.72** | 1.44 | 1.68 | **1.61** | <.001 |
| `rank-avg` (3) | 2.01 | 1.43 | 1.74 | 1.73 | 0.17 |
| **`rank-avg` (5)** † | 1.90 | **1.40** | 1.66 | 1.65 | .001 |
| `rank-avg` (10) | 1.81 | 1.41 | **1.60** | **1.61** | <.001 |
| `max-rank` (3) | 1.99 | 1.42 | 1.73 | 1.71 | 0.12 |
| `max-rank` (5) | 1.80 | 1.43 | 1.67 | 1.63 | <.001 |
| `max-rank` (10) | 1.80 | **1.40** | 1.62 | **1.61** | <.001 |

Bold marks the best value in a column (all rows of a tie). ↑ marks a regression against the same row at the preceding N. † marks the one configuration significant under *all three* evaluators — see point 3 below; the `p` column alone does not show this, since it reports only `fwhisper-lgv3`. Rank ensembles aggregate the cross-family pair {`w2v2-base`, `distil-v3`}.

Three things to read off this table:

1. **`w2v2-base` at N=5 regresses to 2.20% under the official evaluator** (worse than the 2.06% baseline) while *improving* to 1.41% under its own wav2vec 2.0 family. Reported under one evaluator it looks fine; reported under the other it looks harmful. Same audio.
2. **Cross-family ensembles scale monotonically** with N, where the single wav2vec 2.0 verifier does not.
3. **`rank-avg` at N=5 is the only configuration reaching p<0.05 under all three evaluators simultaneously** (p = .001 / .022 / .0002). The two nearest misses both fail under `w2v2-lv60`: `max-rank` N=5 reaches .001 / **.10** / .0001, and `distil-v3` N=10 reaches <.001 / **.18** / .001. Note how little the `fwhisper-lgv3` column tells you here — `max-rank` N=5 beats `rank-avg` N=5 on it (1.80 vs 1.90) and is significant on it, yet is the one that fails the joint bar.

   These are **raw** p-values. `report.py` additionally prints Holm–Bonferroni-corrected decisions within each evaluator column; the `.022` cell above is marginal, and whether it survives that correction depends on the rest of its column. Read the simultaneity result as a descriptive criterion for choosing a configuration, not as a family-wise-error-controlled claim.

## 4. Oracle headroom and recovery (N=3)

The oracle picks the best of 3 per utterance using evaluator scores — not deployable, but it bounds what any verifier could achieve on this candidate pool.

| | fwhisper-lgv3 | w2v2-lv60 | hubert-lg |
|---|---|---|---|
| Single-shot | 2.04 | 1.52 | 1.94 |
| **Oracle (N=3)** | **1.42** | **1.09** | **1.18** |
| Headroom (pp) | 0.63 | 0.43 | 0.76 |

Recovery, `(single − BoN) / (single − oracle)`:

| | fwhisper-lgv3 | w2v2-lv60 | hubert-lg |
|---|---|---|---|
| BoN `w2v2-base` | 7.9 | **26.1** | **27.1** |
| BoN `distil-sm` | 0.0 | 6.8 | 22.6 |
| BoN `distil-v3` | **26.0** | 18.2 | **27.1** |
| BoN `ens3` | 20.5 | 17.0 | 24.5 |

Both tables are computed from unrounded WERs, so recomputing a recovery rate from the rounded values above will be off by a few tenths. Bold marks each evaluator's best verifier (both rows of the `hubert-lg` tie). For the two evaluators that *have* a same-family verifier in this set, the winner is in both cases the verifier **from that evaluator's own family**, on identical audio:

| Evaluator | Same-family verifier | Best cross-family verifier | Ratio |
|---|---|---|---|
| `fwhisper-lgv3` (Whisper) | `distil-v3` — **26.0** | `w2v2-base` — 7.9 | 3.3× |
| `w2v2-lv60` (wav2vec 2.0) | `w2v2-base` — **26.1** | `distil-v3` — 18.2 | 1.4× |

Averaged over these two evaluators, same-family pairs recover **2.0×** what cross-family pairs do (26.1 vs. 13.1). The per-evaluator ratio varies a lot — 3.3× one way, 1.4× the other — so the effect is directional and consistent, not a stable constant. `hubert-lg` is excluded from this comparison because no HuBERT-lineage verifier was run; every entry in its column is cross-family, which is also why all three verifiers land within 22–27% there.

**The counterexample worth stating plainly.** `distil-sm` is Whisper-lineage, so `distil-sm` × `fwhisper-lgv3` is a same-family pair — and it recovers **0.0%**, the worst cell in the table. Family alignment is therefore not sufficient on its own: at 166M `distil-sm` is the weakest verifier in the table under every evaluator — 0.0 / 6.8 / 22.6, worst in each of the three columns — so there is no selection signal for the shared lineage to inflate. The effect we document is conditional on the verifier being competent enough to discriminate in the first place. Pooled over all three same-family pairs the advantage disappears entirely (mean 17.4 vs. 18.3 cross-family) — it is carried by the two competitive verifiers.

Note the single-shot numbers here differ slightly from §3 (2.04 vs. 2.06 under `fwhisper-lgv3`, 1.94 vs. 1.92 under `hubert-lg`; `w2v2-lv60` is unchanged at 1.52). §4 decodes in N=3 batches where §3 decodes one utterance at a time, and batch composition perturbs both Whisper's beam search and the CTC models' padded context. All rows describe the same synthesized audio; the shifts are ~0.02pp and do not move any ranking.

## 5. Quality is preserved

BoN optimizes a content metric, so it could trade timbre or naturalness for WER. It does not:

| Config | SIM-o | UTMOS |
|---|---|---|
| baseline | 0.9426 | 3.8794 |
| BoN `w2v2-base` | 0.9432 | 3.8799 |
| BoN `distil-sm` | 0.9427 | 3.8839 |
| BoN `distil-v3` | 0.9425 | 3.8813 |
| BoN `ens3` | 0.9428 | 3.8795 |

All configurations sit within ±0.001 SIM-o and ±0.005 UTMOS of baseline. These are automatic proxies, not listening tests — read them as "no detected degradation", not as "provably identical quality".

## 6. Why not CKA

The obvious explanation for family alignment would be representational similarity: evaluators with similar audio encoders should agree. The data does not support it. Over the 5 evaluators (10 pairs), higher linear CKA goes with *higher* disagreement in WER rankings, not lower — the opposite of the naive prediction. Two pairs make the point concretely, where agreement is the Pearson correlation of per-configuration WERs:

| Evaluator pair | Linear CKA | Agreement |
|---|---|---|
| `distil-sm` / `whisper-med` | **0.978** | **−0.52** |
| `w2v2-lv60` / `hubert-lg` | 0.545 | **+0.943** |

The nearly representation-identical pair disagrees the most; the representationally distant pair agrees almost perfectly. Across all 10 pairs the CKA-vs-disagreement correlation is +0.36 — **on 10 points that is not statistically significant** (p ≈ 0.3), so treat it as "CKA fails to predict agreement" rather than as evidence of an inverse law.

Encoder geometry is therefore not the mechanism. The pattern is consistent with identity- or lineage-level coupling — a verifier inflating the score of an evaluator that shares its training lineage — the speech analog of LLM-as-a-judge self-preference. Two caveats on that reading. It is descriptive evidence, not a controlled demonstration; parametrically varying evaluator identity would be needed to establish the mechanism. And lineage does not explain everything either: `distil-sm` and `whisper-med` are *both* Whisper-lineage yet are the pair that disagrees most, so shared lineage does not by itself make two systems behave alike. What we claim is the verifier→evaluator asymmetry documented in §4, not a general lineage-similarity law.

## 7. Practical recommendations

- **Report WER under at least two ASR families with disjoint training lineages.** A single-evaluator number cannot distinguish a better verifier from a better-aligned one.
- **Prefer a cross-family rank ensemble** over {`w2v2-base`, `distil-v3`} when robustness matters more than peak single-evaluator WER. Of the two, **`rank_avg` at N=5 is the one to default to** — it is the only configuration significant under all three evaluators (§3). `max_rank` at N=5 gets a better mean and a better official-evaluator WER (1.63 / 1.80 vs. 1.65 / 1.90) but misses the joint bar at `w2v2-lv60` (p=.10); pick it only if you are optimizing the Whisper-family number and are reporting the others anyway.
- **Treat "simultaneously significant under all evaluators" as the bar**, not any single p-value.
- **Use verifiers strong enough to discriminate.** Family alignment only shows up once a verifier can rank candidates at all: `distil-sm` (166M) recovers 0.0–22.6% and buys nothing under any evaluator (§4). Cross-family diversity is a property of a *pool of competent verifiers*, not a substitute for competence.
- **A same-family ensemble should not be expected to help** — two Whisper distillations share the bias that the ensemble is supposed to cancel. This follows from the mechanism rather than from a measured ablation (no same-family ensemble is tabulated here), but `select_candidates.py` warns when the chosen verifiers share a family.
- **Substantial headroom remains.** The best N=3 verifier captures ~26% of the oracle gap, and the oracle keeps falling as N grows (`evaluate_pool.py` + `analyze_oracle.py` recompute it for any N in the pool). Verifier design, not larger N, is where the remaining gain is.

## Caveats

- One base TTS system (F5-TTS) and one corpus (LibriSpeech-PC test-clean). Whether the effect generalizes to CosyVoice 2 / MaskGCT, or to noisier or non-English data, is untested here.
- Quality is measured by automatic SIM-o and UTMOS only; no human listening test was run.
- The CKA analysis is descriptive: 5 evaluators, 10 pairs, and the headline correlation is not significant at that n (§6).
- The family-alignment effect is carried by two verifiers. A third (`distil-sm`) is a same-family pair that shows none of it, and no HuBERT-lineage verifier was run, so one of three evaluator columns cannot test the hypothesis at all.
- These numbers come from the original experiment runs. The code here is a reimplementation validated against their surviving artifacts — see [REPRODUCE.md](REPRODUCE.md#validation-against-the-original-runs) for exactly what was verified and the one known divergence.
