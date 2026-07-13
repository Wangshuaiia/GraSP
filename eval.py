"""Evaluate GraSP predictions against gold answers.

Usage:
    python3 infer.py --num_examples 50 ... > predictions.json
    python3 eval.py --predictions predictions.json
"""

from __future__ import annotations

import argparse
import json
import os

from utils import exact_match, is_refusal, split_gold_answers


def load_predictions(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(predictions: list[dict], skip_refusals: bool = True) -> tuple[dict, list[dict]]:
    num_right = 0
    num_wrong = 0
    num_skipped_refusal = 0
    num_skipped_no_gold = 0
    wrong_examples = []

    for record in predictions:
        prediction = record.get("prediction") or ""
        gold_answers = split_gold_answers(record.get("gold_answer", ""))

        if not gold_answers or gold_answers == ["unknown"]:
            num_skipped_no_gold += 1
            continue
        if skip_refusals and is_refusal(prediction):
            num_skipped_refusal += 1
            continue

        if exact_match(prediction, gold_answers):
            num_right += 1
        else:
            num_wrong += 1
            wrong_examples.append(
                {
                    "qid": record.get("qid"),
                    "question": record.get("question"),
                    "prediction": prediction,
                    "gold_answer": record.get("gold_answer"),
                }
            )

    num_scored = num_right + num_wrong
    metrics = {
        # infer.py produces a single top answer per question (no ranked candidate list), so
        # Hits@1 here is just "is that one answer correct" -- same computation as exact_match(),
        # just the standard KBQA name for it.
        "hits_at_1": num_right / num_scored if num_scored else 0.0,
        "num_right": num_right,
        "num_wrong": num_wrong,
        "num_skipped_refusal": num_skipped_refusal,
        "num_skipped_no_gold": num_skipped_no_gold,
        "num_scored": num_scored,
        "num_total": len(predictions),
    }
    return metrics, wrong_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True, help="Path to infer.py's JSON output")
    parser.add_argument("--output_file", type=str, default=None, help="Where to write the metrics summary (default: <predictions>_eval.json)")
    parser.add_argument(
        "--no_skip_refusals", action="store_true",
        help="Count LLM refusals ('I don't know', etc.) as wrong instead of excluding them from the denominator",
    )
    parser.add_argument("--show_errors", type=int, default=0, help="Print up to N wrong examples for inspection")
    args = parser.parse_args()

    predictions = load_predictions(args.predictions)
    metrics, wrong_examples = evaluate(predictions, skip_refusals=not args.no_skip_refusals)

    print(f"Hits@1: {metrics['hits_at_1']:.4f}")
    print(
        f"right: {metrics['num_right']}, wrong: {metrics['num_wrong']}, "
        f"skipped(refusal): {metrics['num_skipped_refusal']}, skipped(no gold): {metrics['num_skipped_no_gold']}, "
        f"total: {metrics['num_total']}"
    )

    for ex in wrong_examples[: args.show_errors]:
        print(f"\n[{ex['qid']}] Q: {ex['question']}\n  pred: {ex['prediction']}\n  gold: {ex['gold_answer']}")

    output_file = args.output_file or os.path.splitext(args.predictions)[0] + "_eval.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"predictions_file": args.predictions, **metrics}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved metrics to {output_file}")


if __name__ == "__main__":
    main()
