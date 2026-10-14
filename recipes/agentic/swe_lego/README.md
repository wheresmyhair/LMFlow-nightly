# SWE-Lego SFT

Supervised fine-tuning on the [SWE-Lego](https://github.com/SWE-Lego/SWE-Lego) trajectories
([paper](https://arxiv.org/abs/2601.01426)): resolved OpenHands runs on real and synthetic
software-engineering tasks, released as `Lego-X/SWE-Lego-Real-Data` and
`Lego-X/SWE-Lego-Synthetic-Data` (split `resolved`, 5,009 and 9,127 trajectories).

## Data

`convert.py` turns the released rows into an LMFlow `conversation` dataset:

- the leading system message becomes the `system` field;
- user and assistant messages are kept as they are; tool calls stay in the assistant text in
  OpenHands' non-native function-calling format, and observations stay user messages;
- an assistant message with `turn_mask: true` gets `loss: false`. SWE-Lego masks these turns
  (the paper's error masking), so they stay in the context but are not trained on.

```bash
python recipes/agentic/swe_lego/convert.py \
    --source Lego-X/SWE-Lego-Real-Data:resolved \
    --source Lego-X/SWE-Lego-Synthetic-Data:resolved \
    --output_dir data/swe_lego
```

Train with `--conversation_template qwen3_nothink`. It renders every assistant turn verbatim, which
is how SWE-Lego trained (LLaMA-Factory's `qwen3_nothink`) and how the model is served (Qwen3's own
template, without `enable_thinking=false`). Qwen3's own template would add an empty think block to
the last turn only; the train/inference prefix check rejects that.

## Differences from SWE-Lego's LLaMA-Factory fork

On the released data, with the same tokenizer, LMFlow produces the same `input_ids` for every
trajectory that fits in the context, and the same labels except:

- LLaMA-Factory also trains on the `\n` after each `<|im_end|>`; LMFlow stops at `<|im_end|>`,
  the last token the model generates.
- Over-length trajectories: LLaMA-Factory truncates the turn that crosses `cutoff_len` and drops the
  rest; LMFlow ends the sample after the last assistant turn that fits.

## Reference training setup

SWE-Lego-Qwen3-8B (42.2% on SWE-bench Verified, as reported) was trained with Qwen3-8B full
fine-tuning, `cutoff_len` 131072 with YaRN, learning rate 1e-4, 4 epochs, batch 1 per device with
gradient accumulation 8, DeepSpeed ZeRO-3, Liger kernels and offloaded gradient checkpointing
(`examples/train_full/swe_lego_qwen3_8b.yaml` in their repository).

## Smoke test

`smoke.sh` trains Qwen3-0.6B on 8 trajectories on one GPU. It checks that the data loads, renders,
passes the checks and trains; it says nothing about quality.
