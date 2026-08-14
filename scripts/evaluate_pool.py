#!/usr/bin/env python3
"""Transcribe every pooled candidate with one evaluator (for oracle analysis).

Scoring all N candidates under an *evaluator* — not a verifier — is what makes
the oracle computable: the oracle picks the candidate an omniscient verifier
would choose, which bounds what any real verifier can achieve on this pool. The
gap between a strategy and the oracle is the headroom left on the table.

Cost scales with N, so this is the expensive step; ``scripts/analyze_oracle.py``
then reads the cache and needs no GPU.

Usage:
    python scripts/evaluate_pool.py --config ... --evaluator fwhisper-lgv3
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bon_tts.cli import build_parser  # noqa: E402
from bon_tts.config import apply_overrides, load_config  # noqa: E402
from bon_tts.evaluators import EVALUATOR_SPECS, Evaluator, corpus_wer, per_sample_wer  # noqa: E402
from bon_tts.pool import load_pool_records  # noqa: E402
from bon_tts.pool import pool_dir as get_pool_dir  # noqa: E402
from bon_tts.selection import select_oracle  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = build_parser(__doc__)
    parser.add_argument("--config", type=Path, default="configs/librispeech_pc.yaml")
    parser.add_argument("--pool-name", type=str, default=None)
    parser.add_argument("--evaluator", type=str, default="fwhisper-lgv3",
                        help=f"One of {sorted(EVALUATOR_SPECS)}")
    parser.add_argument("--n", type=int, default=None, help="Limit to the first N candidates")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.overrides)
    pool_name = args.pool_name or config["pool"]["name"]
    pool_path = get_pool_dir(config, pool_name)
    records = load_pool_records(pool_path)

    n_candidates = args.n or config["pool"]["n_candidates"]
    eval_dir = pool_path / "candidate_evals"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_dir / f"eval_{args.evaluator}_n{n_candidates}.json"
    if out_path.exists() and not args.overwrite:
        logger.info("%s exists — pass --overwrite to recompute", out_path)
        return

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    logger.info(
        "transcribing %d utterances x %d candidates with %s",
        len(records), n_candidates, args.evaluator,
    )
    evaluator = Evaluator(args.evaluator, device=device)

    entries = []
    for record in tqdm(records, desc=f"Pool eval [{args.evaluator}]"):
        paths = record["candidate_audio"][:n_candidates]
        hyps = evaluator.transcribe(paths, batch_size=args.batch_size)
        reference = record["gen_text"]
        wers = per_sample_wer([reference] * len(hyps), hyps)
        entries.append(
            {
                "idx": record["idx"],
                "reference": reference,
                "candidate_hyps": hyps,
                "candidate_wers": wers,
            }
        )

    references = [e["reference"] for e in entries]
    single_shot = corpus_wer(references, [e["candidate_hyps"][0] for e in entries])
    oracle_hyps = [e["candidate_hyps"][select_oracle(e["candidate_wers"])] for e in entries]
    oracle = corpus_wer(references, oracle_hyps)

    payload = {
        "evaluator": args.evaluator,
        "evaluator_family": evaluator.family,
        "checkpoint": evaluator.checkpoint,
        "normalization": "f5_official",
        "pool_name": pool_name,
        "n_candidates": n_candidates,
        "n_utterances": len(entries),
        "single_shot_corpus_wer": single_shot,
        "oracle_corpus_wer": oracle,
        "per_utterance": entries,
    }
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)

    logger.info(
        "%s: single-shot %.4f  oracle(N=%d) %.4f  headroom %.4f",
        args.evaluator, single_shot, n_candidates, oracle, single_shot - oracle,
    )
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
