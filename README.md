# DAMN for Piano Music Continuation
*Submission for the Symbolic Music Generation task @ MIREX2026*

Inference code for **DAMN (Delta-Absolute Modeling with Nested Transformers)**, submitted to the MIREX 2026 Symbolic Music Generation task.

The system generates a 12-bar piano continuation from the provided symbolic piano prompt.

## Setup

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Install the Hugging Face CLI if needed, then run:

```bash
./setup.sh
```

By default, `setup.sh` downloads the pinned checkpoint from
[`naivecynics/MIREX2026-DAMN`](https://huggingface.co/naivecynics/MIREX2026-DAMN).

## Generation

```bash
./generation.sh /path/to/input.json /path/to/output_directory 8
```

This generates eight continuations:

```text
sample_01.json
...
sample_08.json
```

## Input and output

The input JSON must contain a `prompt` list. Each note has integer `start`, `pitch`, and `duration` fields on the MIREX sixteenth-note grid.

Example:

```json
{
  "prompt": [
    {"start": 16, "pitch": 60, "duration": 4}
  ]
}
```

Each output contains a `generation` list:

```json
{
  "generation": [
    {"start": 80, "pitch": 64, "duration": 4}
  ]
}
```

## MIDI preview

Generated continuations can optionally be rendered with the prompt for inspection:

```bash
python script/preview.py \
  --prompt examples/input.json \
  --generated outputs/sample_01.json \
  --output comparison.mid
```
