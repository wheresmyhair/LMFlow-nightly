#!/usr/bin/env python
"""Convert SWE-Lego trajectories into LMFlow ``conversation`` datasets.

SWE-Lego (https://github.com/SWE-Lego/SWE-Lego) releases OpenHands trajectories as rows of
``{instance_id, messages: [{role, content, turn_mask}], ...}``. The mapping is:

- the leading ``system`` message becomes the instance's ``system`` field;
- ``user`` and ``assistant`` messages are kept in order;
- an assistant message with ``turn_mask: true`` gets ``loss: false``: it stays in the context
  but is not a training target. This matches SWE-Lego's LLaMA-Factory fork
  (``data/processor/supervised.py``), which sets every label of such a turn to IGNORE_INDEX.
  ``turn_mask`` on other roles carries no meaning there and is dropped.

Rows that do not fit this shape are skipped and counted, not repaired.

Example::

    python recipes/agentic/swe_lego/convert.py \
        --source Lego-X/SWE-Lego-Real-Data:resolved \
        --source Lego-X/SWE-Lego-Synthetic-Data:resolved \
        --output_dir data/swe_lego
"""

import argparse
import json
import os
from collections import Counter
from collections.abc import Iterator


def iter_rows(source: str) -> Iterator[dict]:
    """Yield rows from a parquet file, or from a Hugging Face dataset given as ``repo_id:split``."""
    if source.endswith(".parquet"):
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(source).iter_batches(batch_size=64, columns=["instance_id", "messages"]):
            yield from batch.to_pylist()
    else:
        from datasets import load_dataset

        repo_id, _, split = source.partition(":")
        dataset = load_dataset(repo_id, split=split or "resolved").select_columns(["instance_id", "messages"])
        yield from dataset


def convert_row(row: dict) -> tuple[dict | None, str | None]:
    """Map one SWE-Lego row to an LMFlow conversation instance, or return the reason it is skipped."""
    messages = row["messages"]
    if not messages or messages[0]["role"] != "system":
        return None, "no leading system message"
    converted = []
    for index, message in enumerate(messages[1:], start=1):
        role = message["role"]
        if role not in ("user", "assistant"):
            return None, f"unexpected role {role!r} at message {index}"
        item = {"role": role, "content": message["content"]}
        if role == "assistant" and message.get("turn_mask"):
            item["loss"] = False
        converted.append(item)
    if not converted or converted[-1]["role"] != "assistant":
        return None, "does not end with an assistant message"
    return {"conversation_id": row["instance_id"], "system": messages[0]["content"], "messages": converted}, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="A parquet file, or a Hugging Face dataset as repo_id:split. Repeat to concatenate sources.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--shard_size", type=int, default=2000, help="Instances per output JSON file.")
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many instances (0 = all).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    skipped = Counter()
    shard, shard_index, total, masked_turns, assistant_turns = [], 0, 0, 0, 0

    def flush() -> None:
        nonlocal shard, shard_index
        if not shard:
            return
        path = os.path.join(args.output_dir, f"swe_lego_{shard_index:04d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "conversation", "instances": shard}, f, ensure_ascii=False)
        shard, shard_index = [], shard_index + 1

    for source in args.source:
        for row in iter_rows(source):
            instance, reason = convert_row(row)
            if instance is None:
                skipped[reason] += 1
                continue
            assistant = [m for m in instance["messages"] if m["role"] == "assistant"]
            assistant_turns += len(assistant)
            masked_turns += sum(m.get("loss") is False for m in assistant)
            shard.append(instance)
            total += 1
            if len(shard) >= args.shard_size:
                flush()
            if args.limit and total >= args.limit:
                break
        if args.limit and total >= args.limit:
            break
    flush()

    print(
        json.dumps(
            {
                "instances": total,
                "assistant_turns": assistant_turns,
                "assistant_turns_without_loss": masked_turns,
                "skipped": dict(skipped),
                "shards": shard_index,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
