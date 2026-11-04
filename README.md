# DAMN

[![Code](https://img.shields.io/badge/Code-GitHub-181717?style=flat-square&logo=github)](https://github.com/naivecynics/MIREX2026-DAMN)
[![Models](https://img.shields.io/badge/Models-Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/naivecynics/MIREX2026-DAMN)
[![Dataset](https://img.shields.io/badge/Dataset-MetaScore--SFT--100-2563EB?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/naivecynics/MetaScore-SFT-100)
[![Results](https://img.shields.io/badge/MIREX%202026-Results-7C3AED?style=flat-square)](https://music-ir.org/mirex/wiki/2026:Symbolic_Music_Generation_Results)
[![Demo](https://img.shields.io/badge/Demo-Listen-16A34A?style=flat-square)](https://futuremirex.com/demos/symbolic-music-generation-2026/)

**Delta-Absolute Modeling with Nested Transformers** for symbolic piano continuation, submitted to MIREX 2026. Given a four-bar piano prompt with an optional pickup, DAMN generates a twelve-bar continuation.

## MIREX 2026 results

The submitted 144M DAMN achieves the highest mean coherency and structure scores in the official listening evaluation.

| System | Coherency ↑ | Structure ↑ | Creativity ↑ | Musicality ↑ |
| --- | --- | --- | --- | --- |
| DAMN | **3.68 ± 0.16**<sup>a</sup> | **3.39 ± 0.18**<sup>a</sup> | 3.18 ± 0.16<sup>a</sup> | 3.47 ± 0.14<sup>a</sup> |
| PhraseSplitGen | 2.95 ± 0.18<sup>b</sup> | 2.74 ± 0.15<sup>b</sup> | 2.89 ± 0.15<sup>a</sup> | 2.61 ± 0.14<sup>b</sup> |
| MuseCoco (1.2B) | 2.79 ± 0.17<sup>b</sup> | 2.76 ± 0.15<sup>b</sup> | **3.42 ± 0.12**<sup>a</sup> | 2.68 ± 0.16<sup>b</sup> |
| Anticipatory Music Transformer (780M) | 3.47 ± 0.20<sup>a</sup> | 3.34 ± 0.16<sup>a</sup> | 3.34 ± 0.16<sup>a</sup> | **3.61 ± 0.17**<sup>a</sup> |

Mean ± SEM on five-point scales; bold marks the highest mean. Different letters within a column indicate significance at `p < 0.05` under the organizers' Wilcoxon signed-rank tests with Holm–Bonferroni correction.

The study used 8 prompts and 12 listeners, with 38 ratings per system. One continuation was selected from 8 candidates per system and prompt.

[Official results](https://music-ir.org/mirex/wiki/2026:Symbolic_Music_Generation_Results) · [Listening demo](https://futuremirex.com/demos/symbolic-music-generation-2026/)

## Models

| Model | Parameters | Hidden | Score / Note layers | HF directory |
| --- | --- | --- | --- | --- |
| DAMN | 144M | 768 | 14 / 2 | Root |
| DAMN-Mini | 20.5M | 384 | 8 / 1 | `mini/` |

Both models use delta-onset tokens, absolute-onset RoPE, and start EOS. The `mirex2026-submission` tag identifies the challenge submission.

## Setup

```bash
python -m pip install -r requirements.txt
./setup.sh
```

`setup.sh` downloads the pinned 144M checkpoint. To download Mini:

```bash
hf download naivecynics/MIREX2026-DAMN mini/model.safetensors mini/config.json \
  --revision 32390ad4bf8241bbe1ba6c0d3153d2ee389ae80e --local-dir model
```

## Generation

```bash
./generation.sh input.json outputs/damn 8

# Mini
DAMN_MODEL_DIR="$PWD/model/mini" ./generation.sh input.json outputs/mini 8
```

Outputs are `sample_01.json` through `sample_08.json`. Sampling uses temperature 1.0, top-p 0.95, and seed 42.

Input and output notes use integer `(start, pitch, duration)` on a sixteenth-note grid:

```json
{"prompt": [{"start": 16, "pitch": 60, "duration": 4}]}
```

```json
{"generation": [{"start": 80, "pitch": 64, "duration": 4}]}
```

Prompt starts are in `0–79`; continuation starts are in `80–271`. The window ends at step 272, with steps `0–15` reserved for pickup.

Create a multitrack MIDI comparison:

```bash
python scripts/preview.py --prompt input.json \
  --generated outputs/damn/sample_01.json outputs/mini/sample_01.json \
  --output comparison.mid
```

## Training

```bash
python -m pip install -r requirements/training.txt
wandb login
```

Prepared Parquet shards require these columns:

| Column | Contents |
| --- | --- |
| `slice_id` | Sample identifier |
| `dataset_split` | `train`, `validation`, or `test` |
| `notes` | `[delta_start, pitch, duration]` triples; up to 512 real notes |
| `absolute_starts` | Exact slice-local onsets, aligned with `notes` |
| `barlines` | Bar boundaries, ending at the slice endpoint |

Notes are ordered by onset and then pitch. Delta-start and duration are saturated at 127. The loader appends EOS, applies dynamic padding, and masks EOS from pitch/duration loss.

Train DAMN with a global batch of 192:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --num_machines 1 --num_processes 4 \
  --mixed_precision bf16 --dynamo_backend no \
  -m training.train --model-size medium \
  --parquet-data-dir /path/to/parquets --base-path ./runs \
  --batch-size 24 --gradient-accumulation-steps 2 \
  --max-steps 210810 --warmup-steps 2000 --anneal-steps 42162 \
  --min-learning-rate 3e-5 --run-name damn-medium
```

Batch size is per GPU. The example covers five epochs of the original 8.1M-window training set. Mini uses `--model-size mini`; its original run used two GPUs with batch 48 and accumulation 2. Training uses AdamW, a warmup–constant–cosine schedule, and W&B logging (`--no-wandb` disables tracking).

Checkpoints are stored under `runs/checkpoints/<run>/`; the latest three are retained. Resume training or export inference assets:

```bash
accelerate launch --num_processes 4 --dynamo_backend no \
  -m training.train --resume runs/checkpoints/damn-medium/step_00005000

python scripts/export_checkpoint.py \
  --checkpoint runs/checkpoints/damn-medium/step_00210810 --output model-export
```

Resume restores model, optimizer, scheduler, and RNG state; the streaming data traversal restarts. Export writes `model.safetensors` and model-only `config.json`.

## Evaluation

```bash
python -m pip install -r requirements/evaluation.txt
```

Generate eight candidates per prompt from the benchmark's `challenge_test/` directory:

```bash
CUDA_VISIBLE_DEVICES=0 python -m evaluation.run_ours \
  --checkpoint model --test-root /path/to/challenge_test \
  --output-root outputs/damn --n-samples 8 --seed 42 \
  --temperature 1.0 --top-p 0.95 --top-k 0
```

The runner loads the model once, derives a seed per prompt/sample, and resumes completed outputs. Results use `<output>/<cid>/sample_XX.json`. `--offset` and `--limit` partition work; `--min-generation-end` controls the early-stop boundary (default 240).

Install [CLaMP3](https://github.com/sanderwood/clamp3) and its symbolic C2 checkpoint in a separate environment. Set the official `code/config.py` weight prefix to `weights_clamp3_c2` and place the checkpoint under `code/`.

```bash
CUDA_VISIBLE_DEVICES=0 python -m evaluation.score_clamp3 \
  --test-root /path/to/challenge_test \
  --candidate damn=outputs/damn --candidate mini=outputs/mini \
  --n-samples 8 --clamp3-root /path/to/clamp3 \
  --clamp3-python /path/to/clamp3-env/bin/python \
  --output-root outputs/clamp3
```

The metric compares continuation-only C2 embeddings by cosine similarity. Scores are averaged across eight samples per prompt, then across prompts. Confidence intervals and paired comparisons bootstrap prompts. MIDI/feature caches support incremental runs.

Results include `piece_metrics.csv`, `prompt_metrics.csv`, `summary.csv`, and `paired_comparisons.csv`.

## Layout and tests

- `src/`: shared model and inference.
- `training/`: configuration, data loading, loss, scheduler, and trainer.
- `evaluation/`: batch generation, CLaMP3 adapter, and metric aggregation.
- `scripts/`: MIDI preview and checkpoint export.

```bash
python -m pip install -r requirements/dev.txt
python -m pytest tests -q
```
