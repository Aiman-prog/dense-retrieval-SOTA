"""Report BGE-M3 token coverage for BRIGHT queries and relevant documents."""

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils.helpers import get_path, load_config  # noqa: E402


def token_lengths(tokenizer, texts):
    return [
        len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
        for text in texts
    ]


def percentage(part, total):
    return 100 * part / total if total else 0.0


def main():
    config = load_config()
    processed = get_path("processed")
    query_limit = config["model"]["query_max_len"]
    passage_limit = config["model"]["passage_max_len"]
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model"])

    all_query_lengths = []
    totals = {"documents": 0, "fitting": 0, "tokens": 0, "retained": 0}

    print("domain,queries_fit,relevant_docs_fit,relevant_tokens_retained")
    for domain in config["evaluation"]["eval_domains"]:
        with open(processed / f"{domain}_queries.jsonl", encoding="utf-8") as handle:
            query_lengths = token_lengths(
                tokenizer,
                [json.loads(line)["query"] for line in handle if line.strip()],
            )
        all_query_lengths.extend(query_lengths)

        with open(processed / f"{domain}_qrels.txt", encoding="utf-8") as handle:
            relevant_ids = {
                parts[2]
                for line in handle
                if len(parts := line.split()) == 4 and int(parts[3]) > 0
            }

        relevant_texts = {}
        with open(processed / f"{domain}_corpus.jsonl", encoding="utf-8") as handle:
            for line in handle:
                document = json.loads(line)
                docid = str(document["docid"])
                if docid in relevant_ids:
                    relevant_texts[docid] = document["text"]

        missing = relevant_ids - relevant_texts.keys()
        if missing:
            raise ValueError(f"{domain}: {len(missing)} relevant documents missing")

        lengths = token_lengths(
            tokenizer, [relevant_texts[docid] for docid in sorted(relevant_ids)]
        )
        fitting = sum(length <= passage_limit for length in lengths)
        retained = sum(min(length, passage_limit) for length in lengths)

        totals["documents"] += len(lengths)
        totals["fitting"] += fitting
        totals["tokens"] += sum(lengths)
        totals["retained"] += retained

        print(
            f"{domain},"
            f"{percentage(sum(n <= query_limit for n in query_lengths), len(query_lengths)):.2f}%,"
            f"{percentage(fitting, len(lengths)):.2f}%,"
            f"{percentage(retained, sum(lengths)):.2f}%"
        )

    print(
        f"overall,"
        f"{percentage(sum(n <= query_limit for n in all_query_lengths), len(all_query_lengths)):.2f}%,"
        f"{percentage(totals['fitting'], totals['documents']):.2f}%,"
        f"{percentage(totals['retained'], totals['tokens']):.2f}%"
    )


if __name__ == "__main__":
    main()
